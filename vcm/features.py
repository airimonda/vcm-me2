"""Log-mel front end as an nn.Module with an ONNX-friendly STFT.

The STFT is a strided conv1d with fixed (non-trainable) DFT weights, Hann
window folded in. Same module is used for training and inside the exported
ONNX graph, so the Pi only needs onnxruntime + numpy.

waveform (B, N) float in [-1, 1]  ->  log-mel (B, n_mels, T)
with T = (N - n_fft) // hop + 1 (no centre padding; 498 frames for 5 s).
Per-utterance instance normalisation (mean/std over the whole map).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """HTK-style triangular filterbank, shape (n_mels, n_fft // 2 + 1)."""
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        lo, c, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        up = (freqs - lo) / max(c - lo, 1e-9)
        down = (hi - freqs) / max(hi - c, 1e-9)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    # area-normalise so each filter has similar peak scale across the range
    fb /= np.maximum(fb.sum(axis=1, keepdims=True), 1e-9) / 2.0
    return fb.astype(np.float32)


class LogMel(nn.Module):
    def __init__(self, sr: int = 16000, n_fft: int = 400, hop: int = 160,
                 n_mels: int = 64, fmin: float = 20.0, fmax: float = 8000.0,
                 eps: float = 1e-6, norm: bool = True):
        super().__init__()
        self.sr, self.n_fft, self.hop, self.n_mels, self.eps, self.norm = sr, n_fft, hop, n_mels, eps, norm
        n_bins = n_fft // 2 + 1
        n = np.arange(n_fft)[None, :]
        k = np.arange(n_bins)[:, None]
        win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)  # periodic Hann
        ang = 2 * np.pi * k * n / n_fft
        cos_w = (np.cos(ang) * win).astype(np.float32)
        sin_w = (-np.sin(ang) * win).astype(np.float32)
        w = np.concatenate([cos_w, sin_w], axis=0)[:, None, :]          # (2*bins, 1, n_fft)
        self.register_buffer("dft", torch.from_numpy(w), persistent=False)
        self.register_buffer("fb", torch.from_numpy(mel_filterbank(sr, n_fft, n_mels, fmin, fmax)),
                             persistent=False)
        self.n_bins = n_bins

    def n_frames(self, n_samples: int) -> int:
        return (n_samples - self.n_fft) // self.hop + 1

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = wav.unsqueeze(1)                                            # (B,1,N)
        spec = torch.nn.functional.conv1d(x, self.dft, stride=self.hop)  # (B,2*bins,T)
        re, im = spec[:, : self.n_bins], spec[:, self.n_bins:]
        power = re * re + im * im                                       # (B,bins,T)
        mel = torch.matmul(self.fb, power)                              # (B,mels,T)
        out = torch.log(mel + self.eps)
        if self.norm:
            mean = out.mean(dim=(1, 2), keepdim=True)
            var = ((out - mean) ** 2).mean(dim=(1, 2), keepdim=True)
            out = (out - mean) / torch.sqrt(var + 1e-5)
        return out
