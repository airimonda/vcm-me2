#!/usr/bin/env python3
"""One-time Spotify login: Authorization Code flow -> config/spotify.json (client id/secret + refresh token).

Run it on the Mac or on the Pi (it needs a browser only to open the authorize URL, not on this machine):

    python scripts/spotify_auth.py --client-id <id> --client-secret <secret>
    python scripts/spotify_auth.py                      # asks for the id/secret (or reads SPOTIFY_CLIENT_ID/_SECRET)

1. It prints the authorize URL (scopes: user-modify-playback-state user-read-playback-state). Open it in any
   browser and approve.
2. Spotify redirects to the redirect URI (default http://127.0.0.1:8888/callback). Nothing needs to be
   listening there: the browser shows a "can't connect" page. Copy the FULL address from the address bar
   (or just the value after code=) and paste it here.
3. The code is exchanged for a refresh token and written to config/spotify.json (mode 600).

The redirect URI must be registered in your app at https://developer.spotify.com/dashboard (Edit settings ->
Redirect URIs) exactly as given to --redirect-uri.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import secrets
import sys
import urllib.parse
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
SCOPES = "user-modify-playback-state user-read-playback-state"
DEFAULT_REDIRECT = "http://127.0.0.1:8888/callback"


def authorize_url(client_id: str, redirect_uri: str, state: str) -> str:
    q = urllib.parse.urlencode({"client_id": client_id, "response_type": "code", "redirect_uri": redirect_uri,
                                "scope": SCOPES, "state": state, "show_dialog": "true"})
    return f"{AUTH_URL}?{q}"


def extract_code(pasted: str, expected_state: str | None = None) -> str:
    """accepts the full redirected URL, a query string, or the bare code"""
    pasted = pasted.strip()
    if "code=" in pasted or pasted.startswith("http"):
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(pasted).query or pasted.lstrip("?"))
        if "error" in qs:
            raise SystemExit(f"Spotify returned an error: {qs['error'][0]}")
        if expected_state and qs.get("state") and qs["state"][0] != expected_state:
            raise SystemExit("state mismatch: this is not the redirect for the URL printed above")
        if "code" not in qs:
            raise SystemExit("no code= found in what you pasted")
        return qs["code"][0]
    return pasted


def exchange(code: str, client_id: str, client_secret: str, redirect_uri: str) -> dict:
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    r = requests.post(TOKEN_URL, data={"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri},
                      headers={"Authorization": f"Basic {basic}"}, timeout=15)
    if r.status_code != 200:
        raise SystemExit(f"token exchange failed ({r.status_code}): {r.text}")
    return r.json()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--client-id", default=os.environ.get("SPOTIFY_CLIENT_ID"))
    ap.add_argument("--client-secret", default=os.environ.get("SPOTIFY_CLIENT_SECRET"))
    ap.add_argument("--redirect-uri", default=DEFAULT_REDIRECT)
    ap.add_argument("--device-name", default="Watson", help="Spotify Connect name of the Pi (LIBRESPOT_NAME)")
    ap.add_argument("--context-uri", default=None, help="playlist/album to start when nothing is queued")
    ap.add_argument("--out", default=str(REPO_ROOT / "config" / "spotify.json"))
    ap.add_argument("--code", default=None, help="skip the prompt: the code or redirected URL")
    a = ap.parse_args()

    client_id = a.client_id or input("Spotify client id: ").strip()
    client_secret = a.client_secret or getpass.getpass("Spotify client secret: ").strip()
    state = secrets.token_urlsafe(8)
    print("\nOpen this URL in a browser and approve access:\n")
    print(authorize_url(client_id, a.redirect_uri, state))
    print(f"\nThen copy the full address you are redirected to ({a.redirect_uri}?code=...) and paste it here.")
    pasted = a.code or input("\nRedirected URL (or code): ")
    code = extract_code(pasted, state)
    tok = exchange(code, client_id, client_secret, a.redirect_uri)
    if not tok.get("refresh_token"):
        raise SystemExit(f"no refresh_token in the response: {tok}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update({"client_id": client_id, "client_secret": client_secret, "refresh_token": tok["refresh_token"],
                     "redirect_uri": a.redirect_uri, "device_name": a.device_name})
    if a.context_uri:
        existing["context_uri"] = a.context_uri
    existing.setdefault("shuffle", True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(out)
    print(f"\nwrote {out} (scopes: {tok.get('scope', SCOPES)})")
    print("Next: install raspotify on the Pi, name it, then say 'Watson, play music'. See docs/runtime.md.")


if __name__ == "__main__":
    sys.exit(main())
