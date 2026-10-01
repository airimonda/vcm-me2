"""Packed-split dataset (int16 memmap + meta parquet written by scripts/pack_data.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_COLS = ["cmd_idx", "slot_head_idx", "slot_value_idx", "variation_idx"]
WINDOW = 80_000


class PackedSplit(Dataset):
    """Item: (int16 waveform tensor (L,), labels int64 (4,)). L is 96000 for train, 80000 else.
    max_clips takes a seeded random subset (used for smoke runs)."""

    def __init__(self, pack_dir: str | Path, split: str, max_clips: int | None = None, seed: int = 0):
        pack_dir = Path(pack_dir)
        self.split = split
        self.audio = np.load(pack_dir / f"{split}_audio.npy", mmap_mode="r")
        meta = pd.read_parquet(pack_dir / f"{split}_meta.parquet")
        idx = np.arange(len(meta))
        if max_clips is not None and max_clips < len(idx):
            idx = np.sort(np.random.RandomState(seed).choice(idx, max_clips, replace=False))
        self.idx = idx
        self.meta = meta.iloc[idx].reset_index(drop=True)
        self.labels = torch.from_numpy(self.meta[LABEL_COLS].to_numpy().astype(np.int64))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return torch.from_numpy(np.array(self.audio[self.idx[i]])), self.labels[i]

    def sample_weights(self, real_weight: float = 3.0) -> torch.Tensor:
        """Real voices (is_synthetic == 0) are drawn `real_weight` times as often."""
        w = np.where(self.meta["is_synthetic"].to_numpy() == 0, real_weight, 1.0)
        return torch.from_numpy(w).double()


def center_window(wav: torch.Tensor, n: int = WINDOW) -> torch.Tensor:
    """Un-shifted 5 s window from a (B, L) buffer."""
    s = (wav.shape[1] - n) // 2
    return wav[:, s: s + n]


def to_float(wav_i16: torch.Tensor) -> torch.Tensor:
    return wav_i16.float() / 32768.0
