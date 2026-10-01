"""Label space of the Voice Command Model.

* 19 commands + OUT_OF_SCOPE = 20 command classes.
* 6 slotted commands, each with exactly 3 slot values -> 6 slot heads x 3.
* 93 "variations" (Option B phrasings, from variations.csv) + OUT_OF_SCOPE
  = 94 evaluation groups. The model does not predict the variation itself,
  only (command, slot); a clip is right if command and slot are right.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

COMMANDS = [
    "PLAY_MUSIC", "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN",
    "LIGHT_ON", "LIGHT_OFF", "WEATHER", "TIME", "CALL", "MESSAGE",
    "LIST_REMINDERS", "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS",
    "COLOR", "CREATE_REMINDER",
]
OOS = "OUT_OF_SCOPE"
CLASSES = COMMANDS + [OOS]          # 20 command classes
N_CMD = len(CLASSES)
OOS_IDX = N_CMD - 1

SLOT_VALUES = {
    "TIMER": ["10 seconds", "30 seconds", "1 minute"],
    "ALARM": ["6:00 AM", "8:00 AM", "9:00 PM"],
    "TEMPERATURE": ["18 degrees", "22 degrees", "26 degrees"],
    "BRIGHTNESS": ["20 percent", "60 percent", "100 percent"],
    "COLOR": ["Red", "Blue", "Green"],
    "CREATE_REMINDER": ["Drink water", "Study", "Exercise"],
}
SLOT_COMMANDS = list(SLOT_VALUES)                  # head index = position here
N_SLOT_HEADS = len(SLOT_COMMANDS)                  # 6
N_SLOT_VALUES = 3
SLOT_HEAD_OF_CMD = {COMMANDS.index(c): h for h, c in enumerate(SLOT_COMMANDS)}
CMD_OF_SLOT_HEAD = [COMMANDS.index(c) for c in SLOT_COMMANDS]

VARIATIONS_CSV = Path(__file__).resolve().parent.parent / "configs" / "variations.csv"


def _norm(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    return " ".join(str(s).strip().lower().split())


# slot string (normalised) -> value idx, per head
_SLOT_LOOKUP = [
    {_norm(v): i for i, v in enumerate(SLOT_VALUES[c])} for c in SLOT_COMMANDS
]


def load_variations(path: str | Path = VARIATIONS_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def _build_variation_index(path=VARIATIONS_CSV):
    v = load_variations(path)
    by_phrase = {}
    names = []
    for i, r in enumerate(v.itertuples(index=False)):
        key = _norm(r.phrase)
        assert key not in by_phrase, f"duplicate phrase {r.phrase}"
        by_phrase[key] = i
        names.append(str(r.phrase))
    return by_phrase, names


VARIATION_BY_PHRASE, VARIATION_NAMES = _build_variation_index()
N_VARIATIONS = len(VARIATION_NAMES)                # 93
N_EVAL_GROUPS = N_VARIATIONS + 1                   # + OOS group
OOS_GROUP = N_VARIATIONS


def slot_value_idx(command: str, slot_value) -> int:
    """Index of the slot value inside its head, or -1 (not slotted / other value)."""
    if command not in SLOT_VALUES:
        return -1
    return _SLOT_LOOKUP[SLOT_COMMANDS.index(command)].get(_norm(slot_value), -1)


def row_to_labels(row) -> tuple[int, int, int, int]:
    """Map a manifest row (mapping or Series) to
    (cmd_idx, slot_head_idx or -1, slot_value_idx or -1, variation_idx).

    * OOS rows: (19, -1, -1, 93).
    * Non-slotted command: (cmd, -1, -1, variation).
    * Slotted command with one of the 3 values: (cmd, head, value, variation).
    * Slotted command with some other value ("other slot value" bucket):
      (cmd, head, -1, -1); the slot loss is masked, no variation group.
    """
    command = str(row["command"]).strip()
    if command == OOS or int(row.get("out_of_scope", 0) or 0) == 1:
        return OOS_IDX, -1, -1, OOS_GROUP
    cmd = COMMANDS.index(command)
    head = SLOT_HEAD_OF_CMD.get(cmd, -1)
    val = slot_value_idx(command, row.get("slot_value", ""))
    var = VARIATION_BY_PHRASE.get(_norm(row.get("variation", "")), -1)
    return cmd, head, val, var


def variation_to_labels(label: str, value) -> tuple[int, int, int]:
    """(cmd_idx, slot_head, slot_value) for a variations.csv row."""
    cmd = COMMANDS.index(label)
    return cmd, SLOT_HEAD_OF_CMD.get(cmd, -1), slot_value_idx(label, value)


def joint_class(cmd: int, slot: int) -> int:
    """Class index in the 32-way (command, slot) space used for confusion matrices."""
    if cmd == OOS_IDX:
        return 31
    return _JOINT_OFFSET[cmd] + (slot if cmd in SLOT_HEAD_OF_CMD and slot >= 0 else 0)


def _joint_tables():
    off, names, k = {}, [], 0
    for c, name in enumerate(COMMANDS):
        off[c] = k
        if c in SLOT_HEAD_OF_CMD:
            for v in SLOT_VALUES[name]:
                names.append(f"{name}:{v}")
            k += 3
        else:
            names.append(name)
            k += 1
    names.append(OOS)
    return off, names


_JOINT_OFFSET, JOINT_NAMES = _joint_tables()
assert len(JOINT_NAMES) == 32
