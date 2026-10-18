"""3-day sequence dataset over the cached 2-D MJO anomaly maps.

A sample is a short "video" of ``seq_len`` consecutive days ending on an anchor day
``t``:  frames ``[t-(seq_len-1), ..., t-1, t]`` in chronological order (last frame =
current day).  Shape ``(C, T, H, W)`` so a batch is ``(B, C, T, H, W)`` — ready for the
Conv3d patch embed.

Longitude is the full 0-357.5°E circle and is **periodic**; the data needs no special
handling (the model supplies the circular positional embedding and the wrap-aware
masks).  Windows that span a calendar gap are dropped, so every frame pair is exactly
one day apart.  Train/val are split by the calendar year of the anchor day ``t``.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MJOSequence(Dataset):
    """Consecutive ``seq_len``-day windows from ``<cache_dir>/anom.npy``."""

    def __init__(self, cache_dir: str, years: list[int], seq_len: int = 3):
        self.cache_dir = cache_dir
        self.seq_len = int(seq_len)
        self.arr = np.load(os.path.join(cache_dir, "anom.npy"), mmap_mode="r")  # [N,C,H,W]
        self.dates = np.load(os.path.join(cache_dir, "dates.npy"))  # datetime64[D]
        assert len(self.arr) == len(self.dates), "anom/dates length mismatch"

        years = set(int(y) for y in years)
        day = self.dates.astype("datetime64[D]").astype("int64")  # days since 1970
        yr = self.dates.astype("datetime64[Y]").astype(int) + 1970

        # anchor index i is valid if the seq_len-day window is gap-free and the anchor
        # year is in `years`
        anchors = []
        L = self.seq_len
        for i in range(L - 1, len(self.dates)):
            if yr[i] not in years:
                continue
            if day[i] - day[i - L + 1] == L - 1:  # consecutive calendar days
                anchors.append(i)
        self.anchors = np.asarray(anchors, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.anchors)

    def _window(self, i: int) -> np.ndarray:
        L = self.seq_len
        frames = np.asarray(self.arr[i - L + 1 : i + 1])  # (T, C, H, W)
        return np.ascontiguousarray(frames.transpose(1, 0, 2, 3))  # (C, T, H, W)

    def __getitem__(self, k: int) -> dict:
        i = int(self.anchors[k])
        x = torch.from_numpy(self._window(i)).float()  # (C, T, H, W)
        anchor = self.dates[i]
        doy = float((anchor - anchor.astype("datetime64[Y]")) / np.timedelta64(1, "D")) + 1.0
        ang = 2.0 * np.pi * doy / 365.25
        return {
            "x": x,
            "date": torch.tensor(anchor.astype("datetime64[D]").astype("int64")),  # days since epoch
            "doy": torch.tensor(doy, dtype=torch.float32),
            "sincos": torch.tensor([np.sin(ang), np.cos(ang)], dtype=torch.float32),
        }

    # -- convenience for downstream (one window per anchor day, all splits) ---- #
    def anchor_dates(self) -> np.ndarray:
        return self.dates[self.anchors]


def build_dataset(cfg_data: dict, split: str, seq_len: int = 3) -> MJOSequence:
    """split: 'train' | 'val' — year ranges come from the ``data:`` config section."""
    y0, y1 = cfg_data[f"{split}_years"]
    return MJOSequence(cfg_data["cache_dir"], list(range(y0, y1 + 1)), seq_len=seq_len)


class SyntheticSequence(Dataset):
    """Eastward-propagating synthetic clips for the smoke test (no real data needed).

    Each sample is a packet of Gaussian convection that moves east at a random speed,
    so a working encoder/predictor should learn *something* propagation-like and the
    pipeline can be exercised on CPU without the preprocessed cache.
    """

    def __init__(self, n: int = 256, shape=(3, 3, 13, 144), seed: int = 0):
        self.n = n
        self.C, self.T, self.H, self.W = shape
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, k: int) -> dict:
        g = np.random.default_rng(k + 1)
        speed = g.uniform(2, 8)  # patches/day eastward
        lon0 = g.uniform(0, self.W)
        lat0 = g.uniform(self.H * 0.3, self.H * 0.7)
        ii, jj = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing="ij")
        x = np.zeros((self.C, self.T, self.H, self.W), np.float32)
        for t in range(self.T):
            cen = (lon0 + speed * t) % self.W
            dl = np.minimum((jj - cen) % self.W, (cen - jj) % self.W)
            blob = np.exp(-(dl**2) / (2 * 12.0**2) - (ii - lat0) ** 2 / (2 * 3.0**2))
            for c in range(self.C):
                x[c, t] = ((-1) ** c) * blob + 0.1 * g.standard_normal((self.H, self.W))
        doy = float(g.integers(1, 366))
        ang = 2.0 * np.pi * doy / 365.25
        return {
            "x": torch.from_numpy(x),
            "date": torch.tensor(np.int64(k)),
            "doy": torch.tensor(doy, dtype=torch.float32),
            "sincos": torch.tensor([np.sin(ang), np.cos(ang)], dtype=torch.float32),
        }
