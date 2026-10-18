"""Phase clustering of the daily V-JEPA embeddings (SOM + k-means).

Pipeline:
  1. z-score the daily embeddings;
  2. cluster into ``k`` phases with k-means and a small 1-D SOM (cyclic-friendly);
  3. order clusters by the longitude of their composite enhanced-convection centre
     (min OLR anomaly), so cluster id increases eastward like the RMM phases;
  4. composite the OLR/U850 anomaly maps per ordered cluster (mirrors the PhysAnchor /
     RMM phase figures);
  5. compare the cluster labels to the classical WH04 RMM phase of the same days
     (active days only) via adjusted Rand index, NMI, and best circular-rotation accuracy.

    python -m jepa.downstream.cluster --ckpt runs/vjepa/best.tar --k 8
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402


# --------------------------------------------------------------------------- #
# A tiny 1-D self-organizing map (no external dependency).
# --------------------------------------------------------------------------- #
class SOM1D:
    """1-D chain of ``k`` nodes — neighbouring nodes stay similar, good for cyclic phases."""

    def __init__(self, k: int, iters: int = 20000, lr0: float = 0.5,
                 sigma0: float | None = None, seed: int = 0):
        self.k = k
        self.iters = iters
        self.lr0 = lr0
        self.sigma0 = sigma0 if sigma0 is not None else k / 2.0
        self.rng = np.random.default_rng(seed)
        self.W = None

    def fit(self, X: np.ndarray) -> "SOM1D":
        n, d = X.shape
        self.W = X[self.rng.integers(0, n, self.k)].copy()
        pos = np.arange(self.k)
        for it in range(self.iters):
            x = X[self.rng.integers(0, n)]
            bmu = int(np.argmin(((self.W - x) ** 2).sum(1)))
            frac = it / self.iters
            lr = self.lr0 * (1 - frac)
            sigma = max(self.sigma0 * (1 - frac), 0.5)
            h = np.exp(-((pos - bmu) ** 2) / (2 * sigma**2))[:, None]
            self.W += lr * h * (x - self.W)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        d2 = ((X[:, None, :] - self.W[None, :, :]) ** 2).sum(2)
        return d2.argmin(1)


# --------------------------------------------------------------------------- #
def load_inputs(ckpt_or_dir: str, cfg: dict):
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


def order_by_convection(labels: np.ndarray, k: int, emb_dates, anom, anom_dates, stats):
    """Order clusters eastward by the longitude of the composite OLR-anomaly minimum."""
    lon = stats["lon"]
    olr_mean, olr_std = float(stats["mean"][0]), float(stats["std"][0])
    pos = pd.Index(anom_dates).get_indexer(emb_dates)
    centre_lon = np.full(k, np.nan)
    for c in range(k):
        rows = pos[(labels == c) & (pos >= 0)]
        if rows.size == 0:
            continue
        comp = np.asarray(anom[rows, 0]).mean(0) * olr_std + olr_mean  # OLR composite (W/m^2)
        zonal = comp.mean(0)  # meridional mean -> longitude profile
        centre_lon[c] = lon[int(np.argmin(zonal))]  # enhanced convection = min OLR
    order = np.argsort(np.nan_to_num(centre_lon, nan=1e9))
    remap = np.full(k, -1)
    for new, old in enumerate(order):
        remap[old] = new
    return remap[labels], centre_lon[order], order


def circular_accuracy(pred: np.ndarray, ref: np.ndarray, k: int) -> float:
    """Best exact-match fraction over all k rotations + reflection (both are cyclic)."""
    best = 0.0
    for refl in (pred, (k - 1 - pred)):
        for r in range(k):
            best = max(best, np.mean(((refl + r) % k) == ref))
    return best


def run(ckpt: str, k: int = 8, som_iters: int = 30000, save_fig: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
    out_dir, emb, emb_dates, anom, anom_dates, stats, rmm = load_inputs(ckpt, cfg)

    Z = (emb - emb.mean(0)) / (emb.std(0) + 1e-6)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
    som = SOM1D(k, iters=som_iters, seed=0).fit(Z)
    lab_km = km.labels_
    lab_som = som.predict(Z)

    lab_km, cen_km, _ = order_by_convection(lab_km, k, emb_dates, anom, anom_dates, stats)
    lab_som, cen_som, _ = order_by_convection(lab_som, k, emb_dates, anom, anom_dates, stats)

    # align to RMM (active days)
    rmm_by_date = rmm.set_index("date")
    common = np.intersect1d(emb_dates, rmm_by_date.index.values)
    e_idx = pd.Index(emb_dates).get_indexer(common)
    amp = rmm_by_date.loc[common, "amplitude"].values
    rphase = rmm_by_date.loc[common, "phase"].values.astype(int) - 1  # 0..k-1 (k=8)
    active = amp >= 1.0

    results = {"k": k}
    for name, lab in [("kmeans", lab_km), ("som", lab_som)]:
        c = lab[e_idx][active]
        r = rphase[active]
        results[name] = {
            "ARI": float(adjusted_rand_score(r, c)),
            "NMI": float(normalized_mutual_info_score(r, c)),
            "circular_acc": float(circular_accuracy(c, r, k)),
            "n_active": int(active.sum()),
        }

    if save_fig:
        lon, lat = stats["lon"], stats["lat"]
        omean, ostd = float(stats["mean"][0]), float(stats["std"][0])
        umean, ustd = float(stats["mean"][1]), float(stats["std"][1])
        pos = pd.Index(anom_dates).get_indexer(emb_dates)
        nrow = (k + 1) // 2
        fig, axes = plt.subplots(nrow, 2, figsize=(11, 1.7 * nrow), sharex=True, sharey=True)
        for c, ax in zip(range(k), axes.ravel()):
            rows = pos[(lab_km == c) & (pos >= 0)]
            comp_olr = np.asarray(anom[rows, 0]).mean(0) * ostd + omean
            comp_u = np.asarray(anom[rows, 1]).mean(0) * ustd + umean
            pm = ax.pcolormesh(lon, lat, comp_olr, cmap="RdBu_r", vmin=-30, vmax=30, shading="nearest")
            ax.contour(lon, lat, comp_u, levels=[-3, -2, -1, 1, 2, 3], colors="k", linewidths=0.5)
            ax.set_title(f"phase {c+1}  (n={rows.size}, conv≈{cen_km[c]:.0f}°E)", fontsize=8)
        for a in axes[-1]:
            a.set_xlabel("longitude (°E)")
        for a in axes[:, 0]:
            a.set_ylabel("lat")
        fig.colorbar(pm, ax=axes, shrink=0.8, pad=0.02, label="OLR anomaly (W m$^{-2}$)")
        fig.suptitle(f"V-JEPA k-means phase composites (OLR shading, U850 contours)  "
                     f"ARI={results['kmeans']['ARI']:.2f} vs RMM", y=1.0)
        fpath = os.path.join(out_dir, "phase_composites.png")
        fig.savefig(fpath, dpi=120, bbox_inches="tight")
        plt.close(fig)
        results["figure"] = fpath

    # save labels for downstream / inspection
    np.savez(os.path.join(out_dir, "phase_labels.npz"),
             dates=emb_dates, kmeans=lab_km, som=lab_som)
    print("[cluster] agreement with classical RMM phase (active days):")
    for name in ("kmeans", "som"):
        m = results[name]
        print(f"  {name:7s} ARI={m['ARI']:.3f}  NMI={m['NMI']:.3f}  "
              f"circular_acc={m['circular_acc']:.3f}  (n_active={m['n_active']})")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--som_iters", type=int, default=30000)
    args = ap.parse_args()
    run(args.ckpt, k=args.k, som_iters=args.som_iters)


if __name__ == "__main__":
    main()
