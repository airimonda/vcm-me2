import math

import numpy as np
import pytest
import torch

from vcm.augment import PRESETS, Augmenter, NoiseBank, active_power, pitch_shift
from vcm.features import LogMel


def _tone(f, n=96000, amp=0.3):
    t = torch.arange(n) / 16000
    return amp * torch.sin(2 * math.pi * f * t)


def _peak_hz(x):
    sp = torch.fft.rfft(x * torch.hann_window(len(x))).abs()
    return float(sp.argmax()) * 16000 / len(x)


@pytest.mark.parametrize("preset", list(PRESETS))
def test_wave_shapes_and_finite(preset):
    torch.manual_seed(0)
    x = torch.randn(12, 96000) * 0.1
    aug = Augmenter(preset)
    for _ in range(3):
        y = aug.wave(x)
        assert y.shape == (12, 80000)
        assert torch.isfinite(y).all()
        assert y.abs().max() <= 1.0
    f = LogMel()(y)
    g = aug.spec(f)
    assert g.shape == f.shape and torch.isfinite(g).all()


def test_none_is_center_crop():
    x = torch.randn(3, 96000)
    y = Augmenter("none").wave(x)
    assert torch.equal(y, x[:, 8000:88000])


def test_input_not_mutated():
    x = torch.randn(4, 96000) * 0.1
    x0 = x.clone()
    Augmenter("heavy").wave(x)
    assert torch.equal(x, x0)


def test_pitch_shift_frequency():
    x = _tone(440.0)[None]
    for st in (3.0, -3.0):
        y = pitch_shift(x, torch.tensor([st]))
        want = 440 * 2 ** (st / 12)
        assert abs(_peak_hz(y[0]) - want) < 8, (st, _peak_hz(y[0]), want)
        assert y.shape == x.shape


def test_every_op_runs_alone():
    torch.manual_seed(1)
    x = torch.randn(8, 96000) * 0.1
    for op in ["speed", "shift", "pitch", "gain", "polarity", "clip", "eq", "tel", "quant", "noise", "babble", "reverb"]:
        y = Augmenter({op: 1.0}).wave(x)
        assert y.shape == (8, 80000) and torch.isfinite(y).all(), op
        assert not torch.equal(y, x[:, 8000:88000]), op


def test_snr_noise():
    torch.manual_seed(2)
    x = _tone(300.0, 80000, 0.2)[None].repeat(4, 1)
    aug = Augmenter({"noise": 1.0, "snr": (10, 10)})
    y = aug.wave(torch.cat([torch.zeros(4, 8000), x, torch.zeros(4, 8000)], 1))
    noise = y - x
    snr = 10 * torch.log10(active_power(x) / (noise ** 2).mean(1))
    assert (snr - 10).abs().max() < 1.5


def test_noise_bank(tmp_path):
    p = tmp_path / "n.npz"
    np.savez(p, audio=(np.random.randn(50000) * 1000).astype(np.int16), offsets=np.array([0, 20000, 50000]))
    bank = NoiseBank(p)
    s = bank.sample(5, 80000)
    assert s.shape == (5, 80000) and torch.isfinite(s).all()
    y = Augmenter({"noise": 1.0}, bank).wave(torch.randn(6, 96000) * 0.1)
    assert y.shape == (6, 80000)


def test_specaug_masks_and_limits():
    torch.manual_seed(3)
    f = torch.randn(16, 64, 498) + 5
    g = Augmenter({"specaug": 1.0}).spec(f)
    masked = (g == 0)
    assert masked.any()
    # at most 2 time masks of <=40 frames
    assert (masked.all(1).sum(1) <= 80).all()
    # freq masks <= 2*10 bins (a fully-masked row is a freq mask or sits inside time masks)
    assert (masked.all(2).sum(1) <= 20).all()
    g2 = Augmenter({"specaug": 1.0, "timewarp": 1.0}).spec(f)
    assert g2.shape == f.shape and torch.isfinite(g2).all()
