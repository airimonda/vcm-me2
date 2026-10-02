"""Music over the Spotify Web API (plain `requests`, no spotipy).

The Pi runs raspotify (librespot), which shows up as a Spotify Connect device named `music.device_name`
(LIBRESPOT_NAME, default "Watson"). This module drives that device remotely:

    refresh token --> access token (cached, refreshed 60 s early)
    GET  /me/player/devices            find the device by name
    PUT  /me/player                    transfer playback to it (if it is not the active device)
    PUT  /me/player/play               start `context_uri` ("liked" = Liked Songs) or resume it
    PUT  /me/player/pause              PAUSE and STOP (stop = pause)
    POST /me/player/next               NEXT
    PUT  /me/player/volume             VOLUME_UP / VOLUME_DOWN (+-volume_step) and ducking

Credentials (client id/secret + refresh token) are read from config/spotify.json, written once by
scripts/spotify_auth.py. All network calls run on ONE worker thread (blocking `requests`), so ducking,
commands and the now-playing poll are strictly ordered and never block the event loop.

Errors are mapped to a small set of kinds (SpotifyError.kind) with spoken replies:
not_configured, offline, auth, no_device, nothing_queued, premium, api.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from runtime.actuators.base import Actuator
from runtime.command import Response

log = logging.getLogger("spotify")

API = "https://api.spotify.com/v1"
TOKEN_URL = "https://accounts.spotify.com/api/token"

ERROR_REPLY = {
    "not_configured": "music_not_configured",
    "offline": "music_offline",
    "auth": "music_auth",
    "no_device": "music_no_device",
    "nothing_queued": "music_nothing_queued",
    "nothing_playing": "music_nothing_playing",
    "premium": "music_premium",
    "api": "music_error",
    "server": "music_error",
}


class SpotifyError(Exception):
    def __init__(self, kind: str, message: str = ""):
        super().__init__(f"{kind}: {message}" if message else kind)
        self.kind = kind
        self.message = message


def load_credentials(path) -> dict:
    """config/spotify.json -> dict (client_id, client_secret, refresh_token + optional device_name /
    context_uri / shuffle overrides). Missing or incomplete file -> SpotifyError('not_configured')."""
    try:
        creds = json.load(open(path))
    except (OSError, ValueError) as e:
        raise SpotifyError("not_configured", f"cannot read {path}: {e}")
    for k in ("client_id", "client_secret", "refresh_token"):
        if not creds.get(k) or str(creds[k]).startswith("YOUR_"):
            raise SpotifyError("not_configured", f"{path} has no {k}")
    return creds


class SpotifyClient:
    """Synchronous Web API client for one Connect device."""

    def __init__(self, creds: dict, device_name="Watson", context_uri=None, shuffle=True, volume_step=15,
                 timeout=4.0, session=None, creds_path=None, now=time.time, sleep=time.sleep):
        self.creds = dict(creds)
        self.device_name = creds.get("device_name") or device_name
        ctx = creds.get("context_uri") or context_uri
        self.context_uri = None if (ctx and "YOUR_" in ctx) else ctx       # ignore the example file's placeholder
        self.shuffle = creds.get("shuffle", shuffle)
        self.volume_step = int(volume_step)
        self.timeout = timeout
        self.http = session or requests.Session()
        self.creds_path = creds_path
        self._now = now
        self._sleep = sleep
        self._token = None
        self._token_exp = 0.0
        self._device_id = None
        self.last_volume = None            # last volume (%) seen on the device
        self.last_playing = False
        self._tlock = threading.Lock()

    # -- auth ------------------------------------------------------------------
    def access_token(self, force=False) -> str:
        with self._tlock:
            if not force and self._token and self._now() < self._token_exp - 60:
                return self._token
            basic = base64.b64encode(f"{self.creds['client_id']}:{self.creds['client_secret']}".encode()).decode()
            try:
                r = self.http.post(TOKEN_URL, data={"grant_type": "refresh_token",
                                                    "refresh_token": self.creds["refresh_token"]},
                                   headers={"Authorization": f"Basic {basic}"}, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                raise SpotifyError("offline", str(e))
            if r.status_code in (400, 401):
                raise SpotifyError("auth", f"token refresh rejected ({r.status_code}): {r.text[:120]}")
            if r.status_code >= 400:
                raise SpotifyError("api", f"token endpoint {r.status_code}")
            body = r.json()
            self._token = body["access_token"]
            self._token_exp = self._now() + float(body.get("expires_in", 3600))
            new_rt = body.get("refresh_token")
            if new_rt and new_rt != self.creds["refresh_token"]:     # Spotify may rotate it
                self.creds["refresh_token"] = new_rt
                self._persist()
            return self._token

    def _persist(self):
        if not self.creds_path:
            return
        try:
            tmp = str(self.creds_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.creds, f, indent=2)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.creds_path)
        except OSError as e:
            log.warning("could not store the rotated refresh token: %s", e)

    # -- plumbing ----------------------------------------------------------------
    def _request(self, method, path, params=None, body=None, _retry=True):
        """-> parsed JSON dict, or None for an empty (204) reply. Raises SpotifyError."""
        headers = {"Authorization": f"Bearer {self.access_token()}"}
        try:
            r = self.http.request(method, API + path, params=params, json=body, headers=headers,
                                  timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise SpotifyError("offline", str(e))
        sc = r.status_code
        if sc == 401 and _retry:
            self.access_token(force=True)
            return self._request(method, path, params, body, _retry=False)
        if sc == 401:
            raise SpotifyError("auth", "401 after token refresh")
        if sc in (200, 201, 202):
            try:
                return r.json() if r.content else None
            except ValueError:
                return None
        if sc == 204:
            return None
        reason, msg = "", ""
        try:
            err = r.json().get("error", {})
            reason, msg = str(err.get("reason", "")), str(err.get("message", ""))
        except (ValueError, AttributeError):
            pass
        if sc == 403 and "PREMIUM" in reason.upper():
            raise SpotifyError("premium", msg)
        if sc == 403:
            raise SpotifyError("restricted", msg or reason)       # e.g. pausing an already paused player
        if sc == 404:
            raise SpotifyError("no_device", msg or reason or "404")
        if sc == 429:
            raise SpotifyError("api", f"rate limited, retry after {r.headers.get('Retry-After', '?')} s")
        if sc >= 500:                                              # Spotify / Connect device hiccup: retryable
            raise SpotifyError("server", f"{method} {path} -> {sc} {msg}")
        raise SpotifyError("api", f"{method} {path} -> {sc} {msg}")

    # -- devices / state ------------------------------------------------------------
    def devices(self) -> list:
        return (self._request("GET", "/me/player/devices") or {}).get("devices", [])

    def find_device(self, refresh=False) -> dict:
        """the Connect device called `device_name` (exact, case-insensitive; then prefix match)"""
        devs = self.devices()
        want = self.device_name.strip().lower()
        hit = next((d for d in devs if d.get("name", "").strip().lower() == want), None) \
            or next((d for d in devs if d.get("name", "").strip().lower().startswith(want)), None)
        if hit is None:
            self._device_id = None
            names = ", ".join(d.get("name", "?") for d in devs) or "none"
            raise SpotifyError("no_device", f"no Spotify Connect device named {self.device_name!r} (seen: {names})")
        self._device_id = hit["id"]
        return hit

    def playback(self):
        """current playback state dict, or None when nothing is active anywhere"""
        st = self._request("GET", "/me/player", params={"additional_types": "track,episode"})
        if st:
            dev = st.get("device") or {}
            if dev.get("volume_percent") is not None and dev.get("id") == self._device_id:
                self.last_volume = dev["volume_percent"]
            self.last_playing = bool(st.get("is_playing"))
        else:
            self.last_playing = False
        return st

    def snapshot(self, st=None) -> dict:
        """dashboard-shaped view of a playback state (already fetched, or fetched now)"""
        st = st if st is not None else self.playback()
        dev = (st or {}).get("device") or {}
        mine = bool(st) and (dev.get("id") == self._device_id or dev.get("name") == self.device_name)
        item = (st or {}).get("item") if mine else None
        track = None
        if item:
            artists = item.get("artists") or ([{"name": item["show"]["name"]}] if item.get("show") else [])
            track = {"title": item.get("name"), "artist": ", ".join(a.get("name", "") for a in artists) or None,
                     "album": (item.get("album") or {}).get("name")}
        return {"connected": True, "playing": bool(mine and st.get("is_playing")),
                "paused": bool(mine and item and not st.get("is_playing")), "track": track,
                "volume": dev.get("volume_percent") if mine else self.last_volume, "device": self.device_name,
                "error": None}

    def _dev(self) -> str:
        return self._device_id or self.find_device()["id"]

    def _on_device(self, fn):
        """run fn(device_id); on 'device not found' re-discover the device once and retry"""
        try:
            return fn(self._dev())
        except SpotifyError as e:
            if e.kind == "server":                      # transient 5xx: one retry after a moment
                self._sleep(1.0)
                return fn(self._dev())
            if e.kind != "no_device":
                raise
            self._device_id = None
            return fn(self.find_device()["id"])

    # -- commands ----------------------------------------------------------------------
    def resolved_context(self):
        """context_uri, with "liked" meaning the user's Liked Songs (spotify:user:<id>:collection)"""
        if self.context_uri == "liked":
            if not getattr(self, "_liked_uri", None):
                self._liked_uri = "spotify:user:%s:collection" % self._request("GET", "/me")["id"]
            return self._liked_uri
        return self.context_uri

    def play(self) -> dict:
        """Start the configured context (default: Liked Songs), or resume it if it is what is loaded and paused.
        Without a context_uri: resume whatever is loaded. Transfers playback to the device if needed."""
        dev = self.find_device()                      # fresh: is it listed, is it the active device?
        st = self.playback()
        has_item = bool(st and st.get("item"))
        if not dev.get("is_active"):
            self._request("PUT", "/me/player", body={"device_ids": [dev["id"]], "play": False})
        body = None
        ctx = self.resolved_context()
        loaded = ((st or {}).get("context") or {}).get("uri")
        if ctx and not (has_item and loaded == ctx):
            body = {"context_uri": ctx}
            if self.shuffle:
                self._retry_no_device(lambda: self._request("PUT", "/me/player/shuffle",
                                                            params={"state": "true", "device_id": dev["id"]}))
        try:
            self._retry_no_device(lambda: self._request("PUT", "/me/player/play",
                                                        params={"device_id": dev["id"]}, body=body))
        except SpotifyError as e:
            if e.kind in ("no_device", "restricted") and not body:
                raise SpotifyError("nothing_queued", "nothing to resume and no context_uri configured")
            raise
        self.last_playing = True
        return {"connected": True, "playing": True, "paused": False, "device": self.device_name,
                "volume": self.last_volume, "error": None}      # the track shows up on the next refresh

    def _retry_no_device(self, fn, tries=3):
        """after a transfer the device needs a moment before it accepts commands"""
        for i in range(tries):
            try:
                return fn()
            except SpotifyError as e:
                if e.kind not in ("no_device", "server") or i == tries - 1:
                    raise
                self._sleep(0.5 if e.kind == "no_device" else 1.0)   # a just-started librespot needs a moment

    def pause(self) -> dict:
        try:
            self._on_device(lambda d: self._request("PUT", "/me/player/pause", params={"device_id": d}))
        except SpotifyError as e:
            if e.kind == "restricted":                 # already paused
                pass
            elif e.kind == "no_device":
                raise SpotifyError("nothing_playing", "no active playback")
            else:
                raise
        self.last_playing = False
        return {"connected": True, "playing": False, "paused": True, "device": self.device_name,
                "volume": self.last_volume, "error": None}

    def next_track(self) -> dict:
        try:
            self._on_device(lambda d: self._request("POST", "/me/player/next", params={"device_id": d}))
        except SpotifyError as e:
            if e.kind in ("restricted", "no_device"):
                raise SpotifyError("nothing_playing", "nothing to skip")
            raise
        return {"connected": True, "device": self.device_name, "error": None}

    def current_volume(self) -> int:
        if self.last_volume is None:
            self.find_device()
            self.playback()
            if self.last_volume is None:
                dev = next((d for d in self.devices() if d.get("id") == self._device_id), {})
                self.last_volume = dev.get("volume_percent")
        if self.last_volume is None:
            raise SpotifyError("no_device", "cannot read the device volume")
        return int(self.last_volume)

    def set_volume(self, pct: int) -> int:
        pct = max(0, min(100, int(pct)))
        self._on_device(lambda d: self._request("PUT", "/me/player/volume",
                                                params={"volume_percent": pct, "device_id": d}))
        self.last_volume = pct
        return pct


class MusicPlayer:
    """Sync facade the MusicActuator drives: a SpotifyClient (real) or MockPlayer (emulated)."""

    provider = "spotify"


class SpotifyPlayer(MusicPlayer):
    """SpotifyClient + ducking bookkeeping. Every method runs on the worker thread."""

    def __init__(self, client: SpotifyClient, duck_to=20):
        self.client = client
        self.duck_to = duck_to
        self.pre_duck = None
        self.depth = 0
        self.kind = "real"

    # commands (worker thread)
    def play(self):
        return self.client.play()

    def pause(self):
        return self.client.pause()

    def next_track(self):
        return self.client.next_track()

    def change_volume(self, delta):
        """-> (new volume, changed). Applies to the pre-duck level while ducked."""
        base = self.pre_duck if self.pre_duck is not None else self.client.current_volume()
        new = max(0, min(100, base + delta))
        if new == base:
            return base, False
        if self.pre_duck is not None:
            self.pre_duck = new                      # restored at unduck; stay quiet until then
        else:
            self.client.set_volume(new)
        return new, True

    def refresh(self):
        return self.client.snapshot()

    # ducking (worker thread)
    def do_duck(self):
        if self.depth <= 0 or self.pre_duck is not None or not self.client.last_playing:
            return
        cur = self.client.current_volume()
        target = min(cur, self.duck_to)
        if target < cur:
            self.pre_duck = cur
            self.client.set_volume(target)

    def do_unduck(self):
        if self.depth > 0 or self.pre_duck is None:
            return
        pre, self.pre_duck = self.pre_duck, None
        self.client.set_volume(pre)

    def is_playing(self):
        return self.client.last_playing


class MockPlayer(MusicPlayer):
    """In-memory player with the same interface (no account, no network)"""

    TRACKS = [("Mock Song One", "Mock Artist"), ("Mock Song Two", "Mock Artist"), ("Mock Song Three", "Other Artist")]

    def __init__(self, duck_to=20, step=15):
        self.kind = "emulated"
        self.duck_to = duck_to
        self.step = step
        self.i = 0
        self.playing = False
        self.started = False
        self.volume = 50
        self.pre_duck = None
        self.depth = 0
        self.calls = []

    def _snap(self):
        t, a = self.TRACKS[self.i % len(self.TRACKS)]
        return {"connected": True, "playing": self.playing, "paused": self.started and not self.playing,
                "track": {"title": t, "artist": a, "album": "Mock"} if self.started else None,
                "volume": self.pre_duck if self.pre_duck is not None else self.volume, "device": "Mock", "error": None}

    def play(self):
        self.calls.append("play")
        self.playing = self.started = True
        return self._snap()

    def pause(self):
        self.calls.append("pause")
        self.playing = False
        return self._snap()

    def next_track(self):
        self.calls.append("next")
        if not self.started:
            raise SpotifyError("nothing_playing", "nothing to skip")
        self.i += 1
        return self._snap()

    def change_volume(self, delta):
        self.calls.append(f"volume{delta:+d}")
        base = self.pre_duck if self.pre_duck is not None else self.volume
        new = max(0, min(100, base + delta))
        if new == base:
            return base, False
        if self.pre_duck is not None:
            self.pre_duck = new
        else:
            self.volume = new
        return new, True

    def refresh(self):
        return self._snap()

    def do_duck(self):
        if self.depth > 0 and self.pre_duck is None and self.playing and self.volume > self.duck_to:
            self.pre_duck, self.volume = self.volume, self.duck_to

    def do_unduck(self):
        if self.depth == 0 and self.pre_duck is not None:
            self.volume, self.pre_duck = self.pre_duck, None

    def is_playing(self):
        return self.playing


class MusicActuator(Actuator):
    """PLAY_MUSIC / PAUSE / STOP / NEXT / VOLUME_UP / VOLUME_DOWN"""
    name = "music"

    def __init__(self, player: MusicPlayer | None, notify=None, volume_step=15, poll_s=5.0, error: SpotifyError | None = None):
        self.player = player
        self.error = error                       # set when the player could not be built (no credentials)
        self.kind = getattr(player, "kind", "real")
        self.notify = notify or (lambda: None)
        self.volume_step = int(volume_step)
        self.poll_s = poll_s
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spotify")
        self._poll_task = None
        self._state = None
        self._last_err = None

    # -- worker-thread plumbing -----------------------------------------------------
    async def _call(self, fn, *args):
        if self.player is None:
            raise self.error or SpotifyError("not_configured")
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    def _apply(self, state, snap):
        m = state.data["music"]
        m.update(snap)                 # partial snapshots (pause/next) carry no "track": the old one stays
        m["error"] = None
        m["provider"] = getattr(self.player, "provider", "spotify")

    # -- Actuator ---------------------------------------------------------------------
    async def handle(self, cmd, state):
        m = state.data["music"]
        c = cmd.command
        t0 = time.perf_counter()
        self._state = state
        try:
            if self.player is None:
                raise self.error or SpotifyError("not_configured")
            resp = await self._dispatch(c, state)
        except SpotifyError as e:
            log.warning("music %s failed: %s", c, e)
            m["error"] = e.kind
            if e.kind in ("offline", "auth", "not_configured", "no_device"):
                m["connected"] = False
                m["playing"] = False
            self.notify()
            log.warning("spotify %s failed: %s %s", getattr(e, "kind", ""), type(e).__name__, e)
            resp = Response(say=ERROR_REPLY.get(e.kind, "music_error"), ok=False, data={"error": e.kind})
        resp.data["_spotify_start"], resp.data["_spotify_done"] = t0, time.perf_counter()
        return resp

    async def _dispatch(self, c, state):
        m = state.data["music"]
        if c == "PLAY_MUSIC":
            self._apply(state, await self._call(self.player.play))
            m["stopped"] = False
            self.notify()
            self.refresh_soon()
            return Response(say="music_play")
        if c in ("PAUSE", "STOP"):
            self._apply(state, await self._call(self.player.pause))
            if c == "STOP":                                   # stop = pause on Spotify; the dashboard says Stopped
                m["paused"] = False
                m["stopped"] = True
            self.notify()
            return Response(say="music_paused" if c == "PAUSE" else "music_stopped")
        if c == "NEXT":
            snap = await self._call(self.player.next_track)
            self._apply(state, snap)
            self.notify()
            self.refresh_soon()
            return Response(say="music_next")
        if c in ("VOLUME_UP", "VOLUME_DOWN"):
            delta = self.volume_step if c == "VOLUME_UP" else -self.volume_step
            vol, changed = await self._call(self.player.change_volume, delta)
            m["connected"], m["error"], m["volume"] = True, None, vol
            self.notify()
            if not changed:
                return Response(say="music_volume_max" if delta > 0 else "music_volume_min")
            return Response(say="music_volume_up" if delta > 0 else "music_volume_down", data={"n": vol})
        return Response(say=None, ok=False, reason="not_music")

    # -- ducking (called from the event loop, executed in order on the worker thread) -----
    def duck(self):
        if self.player is None:
            return
        self.player.depth += 1
        if self.player.depth == 1:
            self._pool.submit(self._safe, self.player.do_duck)

    def unduck(self):
        if self.player is None or self.player.depth == 0:
            return
        self.player.depth -= 1
        if self.player.depth == 0:
            self._pool.submit(self._safe, self.player.do_unduck)

    def unduck_all(self):
        if self.player is None:
            return
        self.player.depth = 0
        self._pool.submit(self._safe, self.player.do_unduck)

    def _safe(self, fn):
        try:
            fn()
        except Exception as e:                     # ducking must never break a turn
            log.warning("duck/unduck failed: %s", e)

    def is_playing(self):
        return bool(self.player and self.player.is_playing())

    # -- now-playing refresh --------------------------------------------------------------
    def refresh_soon(self, delay=1.0):
        if self.player is None or self._state is None:
            return
        try:
            asyncio.get_running_loop().call_later(delay, lambda: asyncio.ensure_future(self.refresh_once()))
        except RuntimeError:
            pass

    async def refresh_once(self):
        state = self._state
        if state is None or self.player is None:
            return
        m = state.data["music"]
        try:
            snap = await self._call(self.player.refresh)
        except SpotifyError as e:
            if e.kind != self._last_err:
                log.warning("spotify refresh: %s", e)
            self._last_err = e.kind
            if m.get("error") != e.kind or m.get("connected"):
                m["error"], m["connected"], m["playing"] = e.kind, False, False
                self.notify()
            return
        self._last_err = None
        if m.get("stopped") and not snap.get("playing"):
            snap = dict(snap, paused=False)
        before = json.dumps(m, sort_keys=True, default=str)
        if snap.get("playing"):
            m["stopped"] = False
        self._apply(state, snap)
        if json.dumps(m, sort_keys=True, default=str) != before:
            self.notify()

    def start_polling(self, state):
        self._state = state
        if self.poll_s and self.player is not None and self._poll_task is None:
            self._poll_task = asyncio.ensure_future(self._poll_loop())

    async def _poll_loop(self):
        try:
            while True:
                await self.refresh_once()
                await asyncio.sleep(self.poll_s)
        except asyncio.CancelledError:
            pass

    def cancel(self):
        if self._poll_task:
            self._poll_task.cancel()
        if self.player is not None and self.player.depth:
            self.unduck_all()
        self._pool.shutdown(wait=True, cancel_futures=False)

    def status(self):
        return {"name": self.name, "kind": self.kind, "error": getattr(self.error, "kind", None)}


def build_music(cfg: dict, mock: bool, notify=None, repo_root=None, session=None) -> MusicActuator:
    """MusicActuator for config `music:` -- the real Spotify player, MockPlayer when mock, or (no usable
    credentials) an actuator that answers every command with 'Spotify isn't set up yet'."""
    from runtime.config import resolve
    mc = cfg["music"]
    duck_to = cfg["dispatcher"]["duck_to"]
    if mock:
        return MusicActuator(MockPlayer(duck_to=duck_to, step=mc["volume_step"]), notify, mc["volume_step"], poll_s=0)
    path = resolve(mc["credentials"])
    try:
        client = SpotifyClient(load_credentials(path), mc["device_name"], mc["context_uri"], mc["shuffle"],
                               mc["volume_step"], mc["timeout_s"], session=session, creds_path=path)
    except SpotifyError as e:
        log.warning("Spotify not configured (%s); music commands will say so. See docs/runtime.md.", e)
        return MusicActuator(None, notify, mc["volume_step"], poll_s=0, error=e)
    return MusicActuator(SpotifyPlayer(client, duck_to), notify, mc["volume_step"], mc["poll_s"])
