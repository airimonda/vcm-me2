"""Spoken replies: pre-rendered clips by key (scripts/gen_reply_audio.py), espeak-ng / `say` as the fallback.

    text, audio_url = await feedback.say("timer_set", amount="30 seconds")

`resolve(key, values)` computes the deterministic filename gen_reply_audio.py used, so no index file is read
at runtime. A key that is not a template in config/replies.json is treated as literal text (live TTS).
Clip playback never blocks the event loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import shutil
from pathlib import Path

from runtime import sounds
from runtime.config import REPO_ROOT, resolve as resolve_path

log = logging.getLogger("feedback")
EXTS = (".mp3", ".wav", ".ogg")        # mp3 first: mpv, pw-play and afplay all play it


def slug(value):
    s = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return s or "x"


def clip_basename(key, template, values):
    """replies/<this>.mp3 -- shared by the runtime and scripts/gen_reply_audio.py"""
    names = [s["name"] for s in template.get("slots", [])]
    if not names:
        return key
    return "__".join([key] + [f"{n}-{slug(values.get(n))}" for n in names])


def render_text(template, values):
    try:
        text = template["text"].format(**values)
        text = re.sub(r"\b1 (second|minute|hour|reminder)s\b", r"1 \1", text)   # "1 reminders" -> "1 reminder"
    except (KeyError, IndexError, ValueError):
        text = template["text"]
    return text


def load_catalogue(path):
    try:
        data = json.load(open(path))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _play_argv(path):
    if platform.system() == "Darwin":
        return ["afplay", path]
    if shutil.which("mpv"):
        return ["mpv", "--no-video", "--really-quiet", path]
    return ["pw-play" if shutil.which("pw-play") else "aplay", path]     # aplay: wav only


def _tts_argv(text):
    if platform.system() == "Darwin":
        return ["say", text]
    if shutil.which("espeak-ng"):
        return ["espeak-ng", text]
    log.warning("no TTS fallback available (no espeak-ng) for: %s", text)
    return None


class Feedback:
    def __init__(self, replies_dir="replies", catalogue="config/replies.json", bus=None, media=None, play=True,
                 earcon_volume=sounds.DEFAULT_VOLUME):
        self.dir = Path(resolve_path(replies_dir))
        self.templates = load_catalogue(resolve_path(catalogue))
        self.bus = bus
        self.media = media             # object with duck()/unduck() (the music actuator) or None
        self.play = play
        self.earcon_volume = earcon_volume
        self.missing: set[str] = set()

    def resolve(self, key, values):
        """-> (text, path|None)"""
        template = self.templates.get(key)
        if template is None:
            return str(key), None
        text = render_text(template, values)
        base = clip_basename(key, template, values)
        for ext in EXTS:
            p = self.dir / (base + ext)
            if p.exists():
                return text, str(p)
        return text, None

    async def _run(self, argv):
        try:
            return await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.DEVNULL,
                                                        stderr=asyncio.subprocess.DEVNULL)
        except OSError as e:
            log.warning("could not run %s: %s", argv[0], e)
            return None

    async def say(self, key, on_start=None, **values):
        """resolve + play one clip (or its TTS fallback), ducking the music meanwhile; publishes bus "reply".
        Returns (text, audio_url|None)."""
        text, path = self.resolve(key, values)
        if path is None and key in self.templates and (key, tuple(sorted(values.items()))) not in self.missing:
            self.missing.add((key, tuple(sorted(values.items()))))
            log.warning("no reply clip for key=%r values=%r; falling back to TTS", key, values)
        audio_url = "/replies/" + os.path.basename(path) if path else None
        if self.bus is not None:
            self.bus.publish({"type": "reply", "say": text, "key": key, "audio_url": audio_url, "ok": True})
        ducked = False
        if self.media is not None and self.play:
            try:
                self.media.duck()
                ducked = True
            except Exception:
                log.exception("duck failed")
        try:
            proc = None
            if self.play:
                argv = _play_argv(path) if path else _tts_argv(text)
                proc = await self._run(argv) if argv else None
            if on_start is not None:
                on_start()
            if proc is not None:
                await proc.wait()
        finally:
            if ducked:
                try:
                    self.media.unduck()
                except Exception:
                    log.exception("unduck failed")
        return text, audio_url

    async def say_response(self, resp, on_start=None):
        """speak everything in resp.say: one key (+resp.data) or a list of (key, values) pairs.
        Returns (joined_text, [audio_url, ...])."""
        if resp is None or resp.say is None:
            return None, []
        items = resp.say if isinstance(resp.say, list) else [(resp.say, resp.data)]
        texts, urls = [], []
        for key, values in items:
            text, url = await self.say(key, on_start=on_start, **(values or {}))
            on_start = None                      # only the first clip marks the reply start
            texts.append(text)
            if url:
                urls.append(url)
        return " ".join(t for t in texts if t), urls

    def chime(self, name="wake"):
        """short non-blocking earcon"""
        if self.play:
            sounds.play(name, self.earcon_volume)
        if self.bus is not None:
            self.bus.publish({"type": "log", "level": "info", "msg": f"chime:{name}"})
