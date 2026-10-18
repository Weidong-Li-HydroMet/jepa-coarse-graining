"""Shared helpers for the PhysAnchor-MJO-AE downstream science.

Loading the daily latents + the anomaly cache + RMM, building per-phase composites, the
phase "characteristic system center" longitude (mean of the OLR convective centre and the
U850 wind centre), the planetary zonal-wavenumber-1 coherence used to select canonical
phases, and the analysis-domain mask (15°S–15°N, 30°E–210°E — the paper's local sector).

Channel order in the cache is (olr, u850, u200); composites are kept in **standardized**
units (the units the encoder and the daily pattern-correlation both see) and converted to
physical units only for plotting.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Paper analysis sector for the pattern-correlation monitor (Methods, "New Index").
SECTOR_LON = (30.0, 210.0)   # 30°E .. 210°E (Indo-Pacific warm-pool sector)
CH = {"olr": 0, "u850": 1, "u200": 2}


def load_inputs(ckpt_or_dir: str, cfg: dict):
    """Return (out_dir, emb, emb_dates, anom[mmap], anom_dates, stats, rmm)."""
    out_dir = ckpt_or_dir if os.path.isdir(ckpt_or_dir) else os.path.dirname(ckpt_or_dir)
    emb = np.load(os.path.join(out_dir, "embeddings.npy"))
    emb_dates = np.load(os.path.join(out_dir, "emb_dates.npy")).astype("datetime64[D]")

    cache = cfg["data"]["cache_dir"]
    anom = np.load(os.path.join(cache, "anom.npy"), mmap_mode="r")  # [Nall,C,H,W] standardized
    anom_dates = np.load(os.path.join(cache, "dates.npy")).astype("datetime64[D]")
    stats = np.load(os.path.join(cache, "stats.npz"))
    rmm = pd.read_csv(cfg["data"]["rmm_csv"], parse_dates=["time"])
    rmm["date"] = rmm["time"].values.astype("datetime64[D]")
    return out_dir, emb, emb_dates, anom, anom_dates, stats, rmm


def sector_mask(lon: np.ndarray, lo: float = SECTOR_LON[0], hi: float = SECTOR_LON[1]) -> np.ndarray:
    """Boolean longitude mask for the analysis sector (no 0/360° wrap inside 30–210°E)."""
    return (lon >= lo) & (lon <= hi)


def rows_for(emb_dates: np.ndarray, anom_dates: np.ndarray) -> np.ndarray:
    """Row index into the anomaly cache for each embedding date (-1 if absent)."""
    return pd.Index(anom_dates).get_indexer(emb_dates)


def composites(labels: np.ndarray, k: int, emb_dates, anom, anom_dates) -> np.ndarray:
    """Per-phase mean standardized field -> ``(k, C, H, W)`` (NaN-safe for empty phases)."""
    pos = rows_for(emb_dates, anom_dates)
    C, H, W = anom.shape[1:]
    out = np.full((k, C, H, W), np.nan, np.float32)
    for c in range(k):
        rows = pos[(labels == c) & (pos >= 0)]
        if rows.size:
            out[c] = np.asarray(anom[rows]).mean(0)
    return out


def to_physical(comp: np.ndarray, stats) -> np.ndarray:
    """Undo standardization: ``(... ,C, H, W)`` standardized -> physical units."""
    mean = stats["mean"].reshape(-1, 1, 1)
    std = stats["std"].reshape(-1, 1, 1)
    return comp * std + mean


def convective_center_lon(comp_c: np.ndarray, lon: np.ndarray, smask: np.ndarray) -> float:
    """Longitude of enhanced convection = min meridional-mean OLR anomaly within the sector."""
    zonal = comp_c[CH["olr"]].mean(0)  # meridional mean -> (W,)
    j = np.where(smask)[0]
    return float(lon[j[int(np.argmin(zonal[j]))]])


def wind_center_lon(comp_c: np.ndarray, lon: np.ndarray, smask: np.ndarray) -> float:
    """Longitude of the strongest U850 zonal-wind anomaly within the sector."""
    zonal = comp_c[CH["u850"]].mean(0)
    j = np.where(smask)[0]
    return float(lon[j[int(np.argmax(np.abs(zonal[j])))]])


def characteristic_center(comp_c: np.ndarray, lon: np.ndarray, smask: np.ndarray) -> float:
    """Phase characteristic system centre = mean of the OLR and U850 anomaly centres."""
    return 0.5 * (convective_center_lon(comp_c, lon, smask)
                  + wind_center_lon(comp_c, lon, smask))


def wavenumber1_fraction(comp_c: np.ndarray, smask: np.ndarray | None = None) -> float:
    """Fraction of the meridional-mean OLR zonal variance in zonal wavenumber 1–3.

    A proxy for the paper's "coherent, planetary-scale wavenumber-1 structure" criterion:
    canonical MJO phases concentrate power in the lowest zonal wavenumbers; the non-canonical
    SOM nodes do not.  Computed on the FULL longitude circle (FFT needs the periodic field).
    """
    zonal = comp_c[CH["olr"]].mean(0)  # (W,) full circle
    zonal = zonal - zonal.mean()
    power = np.abs(np.fft.rfft(zonal)) ** 2
    if power[1:].sum() <= 0:
        return 0.0
    return float(power[1:4].sum() / power[1:].sum())  # wavenumbers 1..3 / all


def order_by_center(labels: np.ndarray, k: int, comp: np.ndarray,
                    lon: np.ndarray, smask: np.ndarray):
    """Relabel phases 0..k-1 eastward by characteristic-centre longitude.

    Returns (relabelled, centers_sorted, order) where ``order[new]=old``.
    """
    centers = np.array([characteristic_center(comp[c], lon, smask)
                        if np.isfinite(comp[c]).all() else np.nan for c in range(k)])
    order = np.argsort(np.nan_to_num(centers, nan=1e9))
    remap = np.full(k, -1)
    for new, old in enumerate(order):
        remap[old] = new
    return remap[labels], centers[order], order


def circular_accuracy(pred: np.ndarray, ref: np.ndarray, k: int) -> float:
    """Best exact-match fraction over all k rotations + reflection (both cyclic)."""
    best = 0.0
    for refl in (pred, (k - 1 - pred)):
        for r in range(k):
            best = max(best, np.mean(((refl + r) % k) == ref))
    return best
