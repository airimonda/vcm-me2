"""Conversation log: one turn = what the model understood (command + slot, rebuilt as a sentence) + the reply.

There is no transcription, so `heard.text` is rebuilt from command + slot by small templates ("Set a timer
for 30 seconds") and the dashboard labels it "Understood as", never a transcript. Appended to
<run dir>/convo.jsonl, command windows saved to <run dir>/clips/<id>.wav, and pushed to the bus as
{"type": "turn", ...}.
"""
from __future__ import annotations

import json
import os
import time
import wave

import numpy as np


def write_wav(path, audio, sr=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


TEXT = {
    "PLAY_MUSIC": "Play music", "PAUSE": "Pause the music", "STOP": "Stop the music", "NEXT": "Skip to the next song",
    "VOLUME_UP": "Turn the volume up", "VOLUME_DOWN": "Turn the volume down",
    "LIGHT_ON": "Turn the light on", "LIGHT_OFF": "Turn the light off",
    "WEATHER": "What's the weather?", "TIME": "What time is it?",
    "CALL": "Make a call", "MESSAGE": "Send a message", "LIST_REMINDERS": "List my reminders",
    "OUT_OF_SCOPE": "(not a command)",
}
SLOT_TEXT = {
    "TIMER": "Set a timer for {slot}", "ALARM": "Set an alarm for {slot}",
    "TEMPERATURE": "Set the thermostat to {slot}", "BRIGHTNESS": "Set the light to {slot}",
    "COLOR": "Make the light {slot_lower}", "CREATE_REMINDER": "Remind me to {slot_lower}",
}


def rebuild_text(command, slot=None):
    """the "Understood as" sentence for a command + slot"""
    if command in SLOT_TEXT:
        if not slot:
            return command.replace("_", " ").capitalize()
        return SLOT_TEXT[command].format(slot=slot, slot_lower=slot.lower())
    return TEXT.get(command, command)


class ConvoLog:
    def __init__(self, run_dir):
        self.dir = str(run_dir)
        self.path = os.path.join(self.dir, "convo.jsonl")
        self.clips_dir = os.path.join(self.dir, "clips")
        os.makedirs(self.clips_dir, exist_ok=True)
        self._n = 0

    def _next_id(self):
        self._n += 1
        return f"t{int(time.time() * 1000)}_{self._n}"

    def log_turn(self, cmd, accepted, reply_text=None, reply_audio_url=None, audio=None, bus=None, reason=""):
        turn_id = self._next_id()
        clip_url = None
        if audio is not None:
            write_wav(os.path.join(self.clips_dir, f"{turn_id}.wav"), audio)
            clip_url = f"/clips/{turn_id}.wav"
        turn = {
            "id": turn_id, "t": cmd.t,
            "heard": {"text": rebuild_text(cmd.command, cmd.slot), "command": cmd.command, "slot": cmd.slot,
                      "prob": cmd.prob, "raw_command": cmd.raw_command, "accepted": bool(accepted),
                      "audio_url": clip_url, "reason": reason, "source": cmd.source},
            "reply": {"text": reply_text, "audio_url": reply_audio_url} if reply_text else None,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(turn) + "\n")
        if bus is not None:
            bus.publish(dict(type="turn", **turn))
        return turn_id

    def mark(self, turn_id, correct, intended_command=None):
        rec = {"mark": True, "turn_id": turn_id, "correct": bool(correct),
               "intended_command": intended_command, "t": time.time()}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def last(self, n=50):
        if not os.path.exists(self.path):
            return []
        out = []
        for line in open(self.path).readlines()[-n:]:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "heard" in rec:
                out.append(rec)
        return out
