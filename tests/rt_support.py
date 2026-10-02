"""Shared fakes for the runtime tests: a scripted Spotify Web API (token + player endpoints)."""
from __future__ import annotations

import json

import requests


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body
        self.content = b"" if body is None else json.dumps(body).encode()
        self.text = "" if body is None else json.dumps(body)
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class FakeSpotify:
    """Behaves like api.spotify.com/accounts.spotify.com for the endpoints the runtime uses.
    Records every call in `.calls` as (method, path, params, body)."""

    def __init__(self, device_name="Watson", device_active=False, volume=60, with_item=False, premium=True):
        self.device = {"id": "dev123", "name": device_name, "is_active": device_active, "volume_percent": volume,
                       "type": "Speaker"}
        self.devices = [{"id": "phone1", "name": "iPhone", "is_active": False, "volume_percent": 50}, self.device]
        self.item = {"name": "Song A", "artists": [{"name": "Artist A"}], "album": {"name": "Album A"}} if with_item else None
        self.playing = False
        self.premium = premium
        self.calls: list = []
        self.token_posts: list = []
        self.token_expires_in = 3600
        self.token_n = 0
        self.valid_token = None
        self.rotate_refresh = None
        self.offline = False
        self.token_status = 200
        self.force_status: dict = {}          # path -> status to return once

    # ---- requests.Session API --------------------------------------------------
    def post(self, url, data=None, headers=None, timeout=None):
        if self.offline:
            raise requests.ConnectionError("no network")
        self.token_posts.append({"data": dict(data or {}), "auth": (headers or {}).get("Authorization")})
        if self.token_status != 200:
            return FakeResponse(self.token_status, {"error": "invalid_grant"})
        self.token_n += 1
        self.valid_token = f"tok{self.token_n}"
        body = {"access_token": self.valid_token, "token_type": "Bearer", "expires_in": self.token_expires_in}
        if self.rotate_refresh:
            body["refresh_token"] = self.rotate_refresh
        return FakeResponse(200, body)

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):   # noqa: A002
        if self.offline:
            raise requests.ConnectionError("no network")
        path = url.split("/v1", 1)[1]
        self.calls.append((method, path, dict(params or {}), json))
        if (headers or {}).get("Authorization") != f"Bearer {self.valid_token}":
            return FakeResponse(401, {"error": {"status": 401, "message": "The access token expired"}})
        if path in self.force_status:
            return FakeResponse(self.force_status.pop(path), {"error": {"message": "forced"}})
        if path == "/me/player/devices":
            return FakeResponse(200, {"devices": self.devices})
        if path == "/me/player" and method == "GET":
            if not self.device["is_active"]:
                return FakeResponse(204)
            return FakeResponse(200, {"device": self.device, "is_playing": self.playing, "item": self.item})
        if path == "/me/player" and method == "PUT":
            self.device["is_active"] = True
            self.playing = bool(json.get("play")) or self.playing
            return FakeResponse(204)
        if not self.premium and method in ("PUT", "POST"):
            return FakeResponse(403, {"error": {"status": 403, "message": "Player command failed: Premium required",
                                                "reason": "PREMIUM_REQUIRED"}})
        if path == "/me/player/play":
            if not self.device["is_active"]:
                return FakeResponse(404, {"error": {"status": 404, "message": "Device not found", "reason": "NO_ACTIVE_DEVICE"}})
            if json and json.get("context_uri"):
                self.item = {"name": "Playlist Song", "artists": [{"name": "Playlist Artist"}], "album": {"name": "PL"}}
            elif self.item is None:
                return FakeResponse(404, {"error": {"status": 404, "message": "Player command failed: No active device",
                                                    "reason": "NO_ACTIVE_DEVICE"}})
            self.playing = True
            return FakeResponse(204)
        if path == "/me/player/pause":
            if not self.device["is_active"]:
                return FakeResponse(404, {"error": {"status": 404, "message": "no active device", "reason": "NO_ACTIVE_DEVICE"}})
            if not self.playing:
                return FakeResponse(403, {"error": {"status": 403, "message": "Player command failed: Restriction violated",
                                                    "reason": "UNKNOWN"}})
            self.playing = False
            return FakeResponse(204)
        if path == "/me/player/next":
            if not self.device["is_active"]:
                return FakeResponse(404, {"error": {"status": 404, "reason": "NO_ACTIVE_DEVICE"}})
            return FakeResponse(204)
        if path == "/me/player/volume":
            self.device["volume_percent"] = int(params["volume_percent"])
            return FakeResponse(204)
        if path == "/me/player/shuffle":
            return FakeResponse(204)
        return FakeResponse(400, {"error": {"message": f"unhandled {method} {path}"}})

    def paths(self):
        return [(m, p) for m, p, _, _ in self.calls]
