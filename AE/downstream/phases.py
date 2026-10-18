"""Phase anatomy of the daily PhysAnchor-MJO-AE latents (the paper's clustering procedure).

Reproduces Section 1 / "Clustering Procedure" of arXiv:2512.22144:

  1. z-score the daily latents;
  2. **primary clustering = SOM on a 4x2 (8-node) grid** (Methods: the objective
     error-minimization topology);
  3. **select the 6 canonical phases** — the 6 nodes with the most coherent planetary,
     zonal-wavenumber-1 structure; drop the 2 non-canonical nodes (~25% of days);
  4. **validate with linear k-means (k=6)** on the same latents;
  5. order the 6 phases eastward by characteristic-system-centre longitude (mean of the
     OLR convective centre and the U850 wind centre);
  6. composite OLR + U850 anomalies per phase (the Fig. 1 phase maps) and check that the
     two transitional phases — Phase 2 (Indian-Ocean growth) and Phase 4 (Philippine-Sea
     northward shift) — appear;
  7. report SOM<->k-means agreement and agreement with the classical RMM phase.

    python -m AE.downstream.phases --ckpt runs/ae/best.tar
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

from AE.downstream._shared import (  # noqa: E402
    CH, circular_accuracy, composites, load_inputs, order_by_center,
    sector_mask, to_physical, wavenumber1_fraction,
)


# --------------------------------------------------------------------------- #
# A small 2-D self-organizing map (no external dependency) — the paper uses a
# 4x2 SOM grid; node topology is the 2-D lattice, so neighbouring nodes stay similar.
# --------------------------------------------------------------------------- #
class SOM2D:
    def __init__(self, rows: int, cols: int, iters: int = 40000, lr0: float = 0.5,
                 sigma0: float | None = None, seed: int = 0):
        self.rows, self.cols, self.k = rows, cols, rows * cols
        self.iters, self.lr0 = iters, lr0
        self.sigma0 = sigma0 if sigma0 is not None else max(rows, cols) / 2.0
        self.rng = np.random.default_rng(seed)
        # 2-D grid coordinate of each flat node id
        gy, gx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        self.grid = np.stack([gy.ravel(), gx.ravel()], 1).astype(float)  # (k, 2)
        self.W = None

    def fit(self, X: np.ndarray) -> "SOM2D":
        n = X.shape[0]
        self.W = X[self.rng.integers(0, n, self.k)].copy()
        for it in range(self.iters):
            x = X[self.rng.integers(0, n)]
            bmu = int(np.argmin(((self.W - x) ** 2).sum(1)))
            frac = it / self.iters
            lr = self.lr0 * (1 - frac)
            sigma = max(self.sigma0 * (1 - frac), 0.5)
            d2 = ((self.grid - self.grid[bmu]) ** 2).sum(1)
            h = np.exp(-d2 / (2 * sigma**2))[:, None]
            self.W += lr * h * (x - self.W)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        d2 = ((X[:, None, :] - self.W[None, :, :]) ** 2).sum(2)
        return d2.argmin(1)


# --------------------------------------------------------------------------- #
def select_canonical(node_labels: np.ndarray, n_nodes: int, n_keep: int,
                     emb_dates, anom, anom_dates, smask):
    """Keep the ``n_keep`` SOM nodes with the strongest wavenumber-1 OLR structure.

    Returns (canonical_node_ids, w1_fraction_per_node).
    """
    comp = composites(node_labels, n_nodes, emb_dates, anom, anom_dates)
    w1 = np.array([wavenumber1_fraction(comp[c]) if np.isfinite(comp[c]).all() else -1.0
                   for c in range(n_nodes)])
    canonical = np.argsort(w1)[::-1][:n_keep]
    return np.sort(canonical), w1


def rmm_agreement(labels: np.ndarray, emb_dates, rmm, k: int) -> dict:
    """ARI / NMI / circular accuracy vs classical RMM phase on active days (amp>=1)."""
    rmm_by_date = rmm.set_index("date")
    common = np.intersect1d(emb_dates, rmm_by_date.index.values)
    e_idx = pd.Index(emb_dates).get_indexer(common)
    amp = rmm_by_date.loc[common, "amplitude"].values
    rphase8 = rmm_by_date.loc[common, "phase"].values.astype(int) - 1  # 0..7
    lab = labels[e_idx]
    active = (amp >= 1.0) & (lab >= 0)            # drop non-canonical (-1) days
    # map 8 RMM phases -> k bins to compare label partitions at matched resolution
    rphase = (rphase8[active] * k // 8)
    c = lab[active]
    return {
        "ARI": float(adjusted_rand_score(rphase, c)),
        "NMI": float(normalized_mutual_info_score(rphase, c)),
        "circular_acc": float(circular_accuracy(c, rphase, k)),
        "n_active": int(active.sum()),
    }


def run(ckpt: str, n_keep: int = 6, som_rows: int = 4, som_cols: int = 2,
        som_iters: int = 40000, save_fig: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
    out_dir, emb, emb_dates, anom, anom_dates, stats, rmm = load_inputs(ckpt, cfg)
    lon, lat = stats["lon"], stats["lat"]
    smask = sector_mask(lon)
    n_nodes = som_rows * som_cols

    Z = (emb - emb.mean(0)) / (emb.std(0) + 1e-6)

    # ---- 1. primary clustering: 4x2 SOM -> 8 nodes -------------------------- #
    som = SOM2D(som_rows, som_cols, iters=som_iters, seed=0).fit(Z)
    node = som.predict(Z)

    # ---- 2. select the 6 canonical nodes (wavenumber-1 coherence) ----------- #
    canonical, w1 = select_canonical(node, n_nodes, n_keep, emb_dates, anom, anom_dates, smask)
    node_to_phase = {int(nd): i for i, nd in enumerate(canonical)}
    som_lab = np.array([node_to_phase.get(int(n), -1) for n in node])  # -1 = non-canonical
    frac_kept = float((som_lab >= 0).mean())

    # ---- 3. k-means (k=6) validation ---------------------------------------- #
    km = KMeans(n_clusters=n_keep, n_init=10, random_state=0).fit(Z)
    km_lab = km.labels_.copy()

    # ---- 4. order both eastward by characteristic-centre longitude ---------- #
    comp_som = composites(som_lab, n_keep, emb_dates, anom, anom_dates)  # ignores -1
    som_lab_ord, cen_som, _ = order_by_center(
        np.where(som_lab >= 0, som_lab, -1), n_keep, comp_som, lon, smask)
    som_lab_ord = np.where(som_lab >= 0, som_lab_ord, -1)

    comp_km = composites(km_lab, n_keep, emb_dates, anom, anom_dates)
    km_lab_ord, cen_km, _ = order_by_center(km_lab, n_keep, comp_km, lon, smask)

    # recompute ordered composites for plotting / saving
    comp_som = composites(np.where(som_lab_ord >= 0, som_lab_ord, -1), n_keep,
                          emb_dates, anom, anom_dates)
    comp_km = composites(km_lab_ord, n_keep, emb_dates, anom, anom_dates)

    # ---- 5. agreement -------------------------------------------------------- #
    # SOM<->k-means on the canonical days (robustness of the 6-phase taxonomy)
    keep = som_lab_ord >= 0
    som_vs_km = {
        "ARI": float(adjusted_rand_score(som_lab_ord[keep], km_lab_ord[keep])),
        "NMI": float(normalized_mutual_info_score(som_lab_ord[keep], km_lab_ord[keep])),
        "circular_acc": float(circular_accuracy(som_lab_ord[keep], km_lab_ord[keep], n_keep)),
    }
    rmm_som = rmm_agreement(som_lab_ord, emb_dates, rmm, n_keep)
    rmm_km = rmm_agreement(km_lab_ord, emb_dates, rmm, n_keep)

    # eastward-propagation check: characteristic centres must increase with phase id
    east_mono = bool(np.all(np.diff(cen_som) >= -5.0))  # allow tiny non-monotonic wiggle

    results = {
        "k": n_keep, "som_grid": [som_rows, som_cols],
        "frac_days_canonical": frac_kept,
        "som_centers_degE": [round(float(c), 1) for c in cen_som],
        "kmeans_centers_degE": [round(float(c), 1) for c in cen_km],
        "w1_fraction_per_node": [round(float(x), 3) for x in w1],
        "canonical_nodes": canonical.tolist(),
        "som_vs_kmeans": som_vs_km,
        "rmm_agreement_som": rmm_som,
        "rmm_agreement_kmeans": rmm_km,
        "eastward_monotonic": east_mono,
    }

    if save_fig:
        omean, ostd = float(stats["mean"][CH["olr"]]), float(stats["std"][CH["olr"]])
        umean, ustd = float(stats["mean"][CH["u850"]]), float(stats["std"][CH["u850"]])
        names = ["P1", "P2", "P3", "P4", "P5", "P6"]
        fig, axes = plt.subplots(3, 2, figsize=(11, 6), sharex=True, sharey=True)
        for c, ax in zip(range(n_keep), axes.ravel()):
            n_days = int((som_lab_ord == c).sum())
            comp_olr = comp_som[c, CH["olr"]] * ostd + omean
            comp_u = comp_som[c, CH["u850"]] * ustd + umean
            pm = ax.pcolormesh(lon, lat, comp_olr, cmap="RdBu_r", vmin=-30, vmax=30, shading="nearest")
            ax.contour(lon, lat, comp_u, levels=[-3, -2, -1, 1, 2, 3], colors="k", linewidths=0.5)
            ax.axvline(SECTOR := 30, color="0.5", ls=":", lw=0.6); ax.axvline(210, color="0.5", ls=":", lw=0.6)
            ax.set_title(f"{names[c]}  n={n_days}, centre≈{cen_som[c]:.0f}°E", fontsize=9)
        for a in axes[-1]:
            a.set_xlabel("longitude (°E)")
        for a in axes[:, 0]:
            a.set_ylabel("lat")
        fig.colorbar(pm, ax=axes, shrink=0.8, pad=0.02, label="OLR anomaly (W m$^{-2}$)")
        fig.suptitle("PhysAnchor-MJO-AE 6-phase anatomy (SOM 4×2 → 6 canonical; OLR shade, "
                     f"U850 contours)  RMM ARI={rmm_som['ARI']:.2f}", y=1.0)
        fpath = os.path.join(out_dir, "ae_phase_composites.png")
        fig.savefig(fpath, dpi=120, bbox_inches="tight")
        plt.close(fig)
        results["figure"] = fpath

    np.savez(os.path.join(out_dir, "ae_phase_labels.npz"),
             dates=emb_dates, som=som_lab_ord, kmeans=km_lab_ord,
             som_centers=cen_som, kmeans_centers=cen_km,
             canonical_nodes=canonical, w1_fraction=w1,
             composites_som=comp_som, composites_kmeans=comp_km)

    print(f"[ae.phases] SOM {som_rows}x{som_cols}={n_nodes} nodes -> {n_keep} canonical "
          f"(kept {frac_kept*100:.0f}% of days)")
    print(f"[ae.phases] node wavenumber1 fraction: {np.round(w1,2)}  canonical nodes={canonical.tolist()}")
    print(f"[ae.phases] phase centres (°E): SOM {np.round(cen_som,0)}  kmeans {np.round(cen_km,0)}  "
          f"eastward_monotonic={east_mono}")
    print(f"[ae.phases] SOM<->kmeans agreement: ARI={som_vs_km['ARI']:.3f} "
          f"NMI={som_vs_km['NMI']:.3f} acc={som_vs_km['circular_acc']:.3f}")
    print(f"[ae.phases] vs RMM (active days): "
          f"SOM ARI={rmm_som['ARI']:.3f} acc={rmm_som['circular_acc']:.3f} | "
          f"kmeans ARI={rmm_km['ARI']:.3f} acc={rmm_km['circular_acc']:.3f}  "
          f"(n_active≈{rmm_som['n_active']})")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_keep", type=int, default=6)
    ap.add_argument("--som_rows", type=int, default=4)
    ap.add_argument("--som_cols", type=int, default=2)
    ap.add_argument("--som_iters", type=int, default=40000)
    args = ap.parse_args()
    run(args.ckpt, n_keep=args.n_keep, som_rows=args.som_rows,
        som_cols=args.som_cols, som_iters=args.som_iters)


if __name__ == "__main__":
    main()
