"""Decision rule (softmax, tau, slot head) and the sidecar <-> vcm.labels contract."""
import json
from pathlib import Path

import numpy as np
import pytest

from runtime.model import Labels, decide, softmax
from vcm import labels as L

ROOT = Path(__file__).resolve().parent.parent
SIDE = json.loads((ROOT / "models" / "vcm_conformer_M_ens3.json").read_text())
LAB = Labels(SIDE)


def logits_for(cls, margin=8.0, n=20):
    x = np.zeros(n)
    x[L.CLASSES.index(cls)] = margin
    return x


def slot_logits(head_value: dict):
    s = np.zeros((6, 3))
    for cmd, idx in head_value.items():
        s[L.SLOT_COMMANDS.index(cmd), idx] = 6.0
    return s


def test_sidecar_matches_training_label_space():
    assert LAB.classes == L.CLASSES
    assert LAB.slot_commands == L.SLOT_COMMANDS
    assert LAB.slot_values == L.SLOT_VALUES
    assert LAB.tau == pytest.approx(0.40) and len(LAB.commands) == 19


def test_confident_plain_command():
    d = decide(logits_for("PLAY_MUSIC"), slot_logits({}), LAB)
    assert (d.command, d.slot) == ("PLAY_MUSIC", None) and d.prob > 0.99 and not d.is_oos


@pytest.mark.parametrize("cmd", L.SLOT_COMMANDS)
@pytest.mark.parametrize("idx", [0, 1, 2])
def test_slot_is_argmax_of_the_commands_own_head(cmd, idx):
    other = {c: (idx + 1) % 3 for c in L.SLOT_COMMANDS if c != cmd}      # other heads point elsewhere
    d = decide(logits_for(cmd), slot_logits({cmd: idx, **other}), LAB)
    assert d.command == cmd and d.slot == L.SLOT_VALUES[cmd][idx]


def test_non_slot_command_ignores_slot_logits():
    d = decide(logits_for("LIGHT_ON"), slot_logits({"TIMER": 2}), LAB)
    assert d.slot is None


def test_below_tau_is_out_of_scope_and_keeps_the_raw_guess():
    x = np.zeros(20)
    x[L.CLASSES.index("STOP")] = 1.0                  # softmax(STOP) = e/(e+19) ~ 0.125 < 0.40
    d = decide(x, slot_logits({}), LAB)
    assert d.command == "OUT_OF_SCOPE" and d.raw_command == "STOP" and d.slot is None and d.prob < 0.4


def test_tau_boundary_and_override():
    # build logits whose top prob is exactly 0.5
    x = np.zeros(20)
    x[3] = np.log(19.0)                              # p = 19/(19+19) = 0.5
    assert softmax(x)[3] == pytest.approx(0.5)
    assert decide(x, slot_logits({}), LAB).command == L.CLASSES[3]            # tau 0.40 -> accepted
    assert decide(x, slot_logits({}), LAB, tau=0.5 + 1e-9).command == "OUT_OF_SCOPE"
    assert decide(x, slot_logits({}), LAB, tau=0.5).command == L.CLASSES[3]    # prob >= tau passes
    assert decide(x, slot_logits({}), LAB, tau=0.0).command == L.CLASSES[3]


def test_model_says_out_of_scope():
    d = decide(logits_for("OUT_OF_SCOPE"), slot_logits({}), LAB)
    assert d.command == "OUT_OF_SCOPE" and d.is_oos


def test_numerically_stable_for_huge_logits():
    x = np.zeros(20)
    x[5] = 1e4
    d = decide(x, slot_logits({}), LAB)
    assert d.command == L.CLASSES[5] and d.prob == pytest.approx(1.0)
