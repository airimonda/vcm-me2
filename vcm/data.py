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
    max_clips takes a seeded random subset (used for smoke runs).
    indices restricts the split to those row positions of the pack (the tune subset of train / neg_train);
    max_clips is then drawn from that subset. Without indices nothing changes."""

    def __init__(self, pack_dir: str | Path, split: str, max_clips: int | None = None, seed: int = 0,
                 indices=None):
        pack_dir = Path(pack_dir)
        self.split = split
        self.audio = np.load(pack_dir / f"{split}_audio.npy", mmap_mode="r")
        meta = pd.read_parquet(pack_dir / f"{split}_meta.parquet")
        idx = np.arange(len(meta)) if indices is None else np.sort(np.asarray(indices, dtype=np.int64))
        if max_clips is not None and max_clips < len(idx):
            idx = np.sort(np.random.RandomState(seed).choice(idx, max_clips, replace=False))
        self.idx = idx
        self.meta = meta.iloc[idx].reset_index(drop=True)
        self.labels = torch.from_numpy(self.meta[LABEL_COLS].to_numpy().astype(np.int64))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return torch.from_numpy(np.array(self.audio[self.idx[i]])), self.labels[i]

    def sample_weights(self, real_weight: float = 3.0, oos_weight: float = 1.0) -> torch.Tensor:
        """Real voices (is_synthetic == 0) are drawn `real_weight` times as often,
        out-of-scope clips `oos_weight` times as often."""
        w = np.where(self.meta["is_synthetic"].to_numpy() == 0, real_weight, 1.0)
        w = w * np.where(self.meta["command"].to_numpy() == "OUT_OF_SCOPE", oos_weight, 1.0)
        return torch.from_numpy(w).double()


def load_tune_split(path: str | Path) -> dict:
    """tune_split.json written by scripts/make_tune_split.py (row positions in the train / neg_train packs)."""
    import json
    d = json.loads(Path(path).read_text())
    for k in ("tune_train_idx", "train_idx", "neg_tune_idx", "neg_train_idx"):
        d[k] = np.asarray(d[k], dtype=np.int64)
    return d


def center_window(wav: torch.Tensor, n: int = WINDOW) -> torch.Tensor:
    """Un-shifted 5 s window from a (B, L) buffer."""
    s = (wav.shape[1] - n) // 2
    return wav[:, s: s + n]


def to_float(wav_i16: torch.Tensor) -> torch.Tensor:
    return wav_i16.float() / 32768.0
