"""Spotify actuator with a mocked Web API (no network, no credentials)."""
import asyncio
import json

import pytest

from rt_support import FakeSpotify
from runtime.actuators.spotify import (MockPlayer, MusicActuator, SpotifyClient, SpotifyError, SpotifyPlayer,
                                       build_music, load_credentials)
from runtime.command import Command
from runtime.state import State

CREDS = {"client_id": "cid", "client_secret": "sec", "refresh_token": "rt0"}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(fake=None, clock=None, **kw):
    fake = fake or FakeSpotify()
    clock = clock or Clock()
    client = SpotifyClient(dict(CREDS), "Watson", kw.pop("context_uri", None), kw.pop("shuffle", True), 15, 2.0,
                           session=fake, now=clock, sleep=lambda s: None, **kw)
    return fake, clock, client


def run_cmd(act, name, state=None):
    state = state or State()
    return asyncio.run(_run(act, name, state)), state


async def _run(act, name, state):
    return await act.handle(Command(name), state)


# ---------------------------------------------------------------- auth
def test_token_refresh_uses_basic_auth_and_is_cached():
    fake, clock, c = make()
    c.devices()
    c.devices()
    assert len(fake.token_posts) == 1
    post = fake.token_posts[0]
    assert post["data"] == {"grant_type": "refresh_token", "refresh_token": "rt0"}
    import base64
    assert post["auth"] == "Basic " + base64.b64encode(b"cid:sec").decode()


def test_token_refreshed_before_expiry():
    fake, clock, c = make()
    c.devices()
    clock.t += 3600 - 30            # inside the 60 s safety margin
    c.devices()
    assert len(fake.token_posts) == 2


def test_expired_token_401_triggers_one_refresh_and_retry():
    fake, clock, c = make()
    c.devices()
    fake.valid_token = "something-else"      # server revoked it
    assert c.devices()
    assert len(fake.token_posts) == 2


def test_rotated_refresh_token_is_persisted(tmp_path):
    path = tmp_path / "spotify.json"
    path.write_text(json.dumps(CREDS))
    fake = FakeSpotify()
    fake.rotate_refresh = "rt1"
    client = SpotifyClient(load_credentials(path), "Watson", session=fake, creds_path=str(path))
    client.devices()
    assert json.loads(path.read_text())["refresh_token"] == "rt1"


def test_revoked_refresh_token_is_an_auth_error():
    fake, _, c = make()
    fake.token_status = 400
    with pytest.raises(SpotifyError) as e:
        c.devices()
    assert e.value.kind == "auth"


def test_missing_or_placeholder_credentials(tmp_path):
    with pytest.raises(SpotifyError) as e:
        load_credentials(tmp_path / "nope.json")
    assert e.value.kind == "not_configured"
    p = tmp_path / "ex.json"
    p.write_text(json.dumps({"client_id": "YOUR_X", "client_secret": "s", "refresh_token": "r"}))
    with pytest.raises(SpotifyError):
        load_credentials(p)


# ---------------------------------------------------------------- device lookup
def test_find_device_by_name_case_insensitive():
    fake, _, c = make()
    c.device_name = "wATSON"
    assert c.find_device()["id"] == "dev123"


def test_find_device_missing_lists_what_it_saw():
    fake, _, c = make()
    c.device_name = "Kitchen"
    with pytest.raises(SpotifyError) as e:
        c.find_device()
    assert e.value.kind == "no_device" and "iPhone" in str(e.value)


# ---------------------------------------------------------------- commands
def test_play_transfers_then_starts_context_when_nothing_queued():
    fake, _, c = make(context_uri="spotify:playlist:abc", shuffle=True)
    c.play()
    paths = fake.paths()
    assert ("PUT", "/me/player") in paths and paths.index(("PUT", "/me/player")) < paths.index(("PUT", "/me/player/play"))
    transfer = next(call for call in fake.calls if call[:2] == ("PUT", "/me/player"))
    assert transfer[3] == {"device_ids": ["dev123"], "play": False}
    play = next(call for call in fake.calls if call[1] == "/me/player/play")
    assert play[2]["device_id"] == "dev123" and play[3] == {"context_uri": "spotify:playlist:abc"}
    assert ("PUT", "/me/player/shuffle") in paths
    assert fake.playing


def test_play_resumes_when_something_is_queued_without_overriding_it():
    fake = FakeSpotify(device_active=True, with_item=True)
    fake, _, c = make(fake, context_uri="spotify:playlist:abc")
    c.play()
    play = next(call for call in fake.calls if call[1] == "/me/player/play")
    assert play[3] is None                              # plain resume, the configured playlist is NOT restarted
    assert ("PUT", "/me/player") not in fake.paths()    # already the active device: no transfer
    assert fake.playing


def test_play_with_nothing_queued_and_no_context_says_so():
    fake, _, c = make(FakeSpotify(device_active=False, with_item=False))
    with pytest.raises(SpotifyError) as e:
        c.play()
    assert e.value.kind == "nothing_queued"


def test_pause_is_idempotent():
    fake = FakeSpotify(device_active=True, with_item=True)
    fake, _, c = make(fake)
    c.find_device()
    c.pause()                       # not playing -> Spotify answers 403 "Restriction violated": treated as done
    fake.playing = True
    c.pause()
    assert fake.playing is False


def test_volume_set_clamps_and_remembers():
    fake, _, c = make(FakeSpotify(device_active=True, volume=95))
    c.find_device()
    assert c.set_volume(150) == 100
    assert fake.device["volume_percent"] == 100


def run_actuator(fake, steps, mock_player=None, **kw):
    """drive MusicActuator.handle for a list of command names; returns list of Response and the state"""
    _, _, client = make(fake, **kw)
    act = MusicActuator(SpotifyPlayer(client, duck_to=20) if mock_player is None else mock_player, poll_s=0)
    state = State()

    async def go():
        out = []
        for name in steps:
            out.append(await act.handle(Command(name), state))
        return out
    try:
        return asyncio.run(go()), state, act
    finally:
        act._pool.shutdown(wait=True)


def test_actuator_play_pause_next_stop_replies():
    fake = FakeSpotify(device_active=True, with_item=True)
    resps, state, _ = run_actuator(fake, ["PLAY_MUSIC", "NEXT", "PAUSE", "PLAY_MUSIC", "STOP"])
    assert [r.say for r in resps] == ["music_play", "music_next", "music_paused", "music_play", "music_stopped"]
    assert ("POST", "/me/player/next") in fake.paths()
    pauses = [c for c in fake.calls if c[1] == "/me/player/pause"]
    assert len(pauses) == 2                              # PAUSE and STOP both pause (stop = pause)
    assert state.data["music"]["stopped"] is True and state.data["music"]["playing"] is False


def test_actuator_volume_up_down_by_15_percent_with_limits():
    fake = FakeSpotify(device_active=True, volume=60)
    resps, state, _ = run_actuator(fake, ["VOLUME_UP", "VOLUME_UP", "VOLUME_UP", "VOLUME_UP", "VOLUME_UP"])
    vols = [c[2]["volume_percent"] for c in fake.calls if c[1] == "/me/player/volume"]
    assert vols == [75, 90, 100]                         # 60 -> 75 -> 90 -> 100 (clamped)
    assert [r.say for r in resps] == ["music_volume_up"] * 3 + ["music_volume_max"] * 2
    fake2 = FakeSpotify(device_active=True, volume=10)
    resps, state, _ = run_actuator(fake2, ["VOLUME_DOWN", "VOLUME_DOWN"])
    assert [r.say for r in resps] == ["music_volume_down", "music_volume_min"]
    assert fake2.device["volume_percent"] == 0 and state.data["music"]["volume"] == 0


def test_actuator_offline_error_is_spoken():
    fake = FakeSpotify()
    fake.offline = True
    resps, state, _ = run_actuator(fake, ["PLAY_MUSIC", "PAUSE", "VOLUME_UP"])
    assert [r.say for r in resps] == ["music_offline"] * 3
    assert all(not r.ok for r in resps)
    assert state.data["music"]["connected"] is False and state.data["music"]["error"] == "offline"


def test_actuator_device_not_found_and_auth_and_premium():
    f = FakeSpotify(device_name="Other")
    assert run_actuator(f, ["PLAY_MUSIC"])[0][0].say == "music_no_device"
    f = FakeSpotify()
    f.token_status = 400
    assert run_actuator(f, ["PAUSE"])[0][0].say == "music_auth"
    f = FakeSpotify(device_active=True, with_item=True, premium=False)
    assert run_actuator(f, ["NEXT"])[0][0].say == "music_premium"


def test_actuator_nothing_playing():
    fake = FakeSpotify(device_active=False)
    resps, _, _ = run_actuator(fake, ["PAUSE", "NEXT"])
    assert [r.say for r in resps] == ["music_nothing_playing"] * 2


def test_not_configured_actuator_answers_every_command(tmp_path):
    from runtime.config import load_config
    cfg = load_config(None, {"music": {"credentials": str(tmp_path / "missing.json")}})
    act = build_music(cfg, mock=False)
    assert act.player is None
    resp, _ = run_cmd(act, "PLAY_MUSIC")
    assert resp.say == "music_not_configured" and resp.ok is False
    act.cancel()


# ---------------------------------------------------------------- ducking
def test_duck_lowers_volume_and_unduck_restores_it():
    fake = FakeSpotify(device_active=True, volume=70)
    _, _, client = make(fake)
    client.find_device()
    client.last_volume, client.last_playing = 70, True
    act = MusicActuator(SpotifyPlayer(client, duck_to=20), poll_s=0)
    act.duck()
    act.duck()                                    # nested (wake duck + reply duck)
    act._pool.submit(lambda: None).result()       # wait for the worker queue
    assert fake.device["volume_percent"] == 20
    act.unduck()
    act._pool.submit(lambda: None).result()
    assert fake.device["volume_percent"] == 20    # still one outstanding duck
    act.unduck()
    act._pool.submit(lambda: None).result()
    assert fake.device["volume_percent"] == 70
    act._pool.shutdown(wait=True)


def test_duck_does_nothing_when_music_is_not_playing_and_never_raises_volume():
    fake = FakeSpotify(device_active=True, volume=10)
    _, _, client = make(fake)
    client.find_device()
    client.last_volume, client.last_playing = 10, False
    act = MusicActuator(SpotifyPlayer(client, duck_to=20), poll_s=0)
    act.duck()
    act.unduck()
    act._pool.submit(lambda: None).result()
    assert not [c for c in fake.calls if c[1] == "/me/player/volume"]
    client.last_playing = True                   # playing but already quieter than the duck level
    act.duck()
    act.unduck()
    act._pool.submit(lambda: None).result()
    assert fake.device["volume_percent"] == 10
    act._pool.shutdown(wait=True)


def test_volume_command_while_ducked_changes_the_restored_level():
    player = MockPlayer(duck_to=20)
    player.playing, player.started, player.volume = True, True, 60
    act = MusicActuator(player, poll_s=0)
    act.duck()
    act._pool.submit(lambda: None).result()
    assert player.volume == 20 and player.pre_duck == 60
    resp, _ = run_cmd(act, "VOLUME_UP")
    assert resp.say == "music_volume_up" and player.pre_duck == 75
    act.unduck()
    act._pool.submit(lambda: None).result()
    assert player.volume == 75
    act._pool.shutdown(wait=True)


def test_mock_player_end_to_end():
    act = MusicActuator(MockPlayer(), poll_s=0)
    resps = [run_cmd(act, c)[0].say for c in ("NEXT", "PLAY_MUSIC", "NEXT", "PAUSE", "STOP")]
    assert resps == ["music_nothing_playing", "music_play", "music_next", "music_paused", "music_stopped"]
    act._pool.shutdown(wait=True)
