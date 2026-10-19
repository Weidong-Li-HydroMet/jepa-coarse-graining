"""Dataset of daily [3, 32, 360] OLR/U850/U200 anomaly maps + date index.

Reads the per-year arrays produced by `scripts/preprocess_daily.py`
(daily means of the staged 6-hourly 1-degree data, anomalies w.r.t. a smooth
seasonal climatology, per-channel standardized):

    <data_dir>/{year}.npy        float32 [n_days, 3, 32, 360]
    <data_dir>/{year}_doy.npy    int16   [n_days]   day of year (1..366)
    <data_dir>/stats.npz         climatology + mean/std actually used

Labels for the season probe are derived from the date index, they are free:
    sin/cos(2*pi*doy/365), month (0..11), season (DJF/MAM/JJA/SON -> 0..3).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset


def season_of_month(month0: np.ndarray) -> np.ndarray:
    """0-based month -> season id (0=DJF, 1=MAM, 2=JJA, 3=SON)."""
    return ((month0 + 1) % 12) // 3


class MJODaily(Dataset):
    """Daily anomaly maps for the years requested; memory-mapped per year."""

    def __init__(self, data_dir: str, years: list[int]):
        self.data_dir = data_dir
        self.years = list(years)
        self.arrays: list[np.ndarray] = []
        doys, months = [], []
        for y in self.years:
            arr = np.load(os.path.join(data_dir, f"{y}.npy"), mmap_mode="r")
            doy = np.load(os.path.join(data_dir, f"{y}_doy.npy"))
            assert len(arr) == len(doy), f"{y}: data/doy length mismatch"
            self.arrays.append(arr)
            doys.append(doy.astype(np.int32))
            # month from doy (non-leap mapping is fine for labels)
            edges = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
            months.append(np.searchsorted(edges, np.minimum(doy, 365), side="left"))
        self.doy = np.concatenate(doys)
        self.month = np.concatenate(months).astype(np.int64)
        self.season = season_of_month(self.month)
        self._offsets = np.cumsum([0] + [len(a) for a in self.arrays])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, i: int) -> dict:
        k = int(np.searchsorted(self._offsets, i, side="right") - 1)
        x = torch.from_numpy(np.array(self.arrays[k][i - self._offsets[k]]))
        doy = float(self.doy[i])
        ang = 2.0 * np.pi * doy / 365.25
        return {
            "x": x,  # [3, 32, 360] float32
            "doy": torch.tensor(doy, dtype=torch.float32),
            "sincos": torch.tensor([np.sin(ang), np.cos(ang)], dtype=torch.float32),
            "month": torch.tensor(self.month[i]),
            "season": torch.tensor(self.season[i]),
        }


def build_dataset(cfg: dict, split: str) -> MJODaily:
    """split: 'train' | 'val' — year ranges come from the `data:` config section."""
    y0, y1 = cfg[f"{split}_years"]
    return MJODaily(cfg["data_dir"], list(range(y0, y1 + 1)))
