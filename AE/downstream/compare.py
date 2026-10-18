"""AE vs V-JEPA — a like-for-like comparison of the two learned representations.

Because the encoder class/architecture and the latent dimension are identical (the AE reuses
the V-JEPA ``STViTBackbone`` and ``embed_dim``) and both see the same daily clips, the two
daily-latent sets differ ONLY in the training objective (pixel reconstruction vs latent
prediction).  This module clusters each into 6 phases the same way and reports, side by side:

  * clustering quality        — silhouette of the k=6 partition;
  * agreement with RMM phase  — ARI / NMI / circular accuracy on active days;
  * composite coherence       — mean zonal-wavenumber-1 fraction of the 6 phase composites;
  * eastward monotonicity     — do phase centres increase eastward?

Produces one comparison figure (``ae_vs_jepa.png``) and a printed/CSV table.

    python -m AE.downstream.compare --ae_ckpt runs/ae/best.tar --jepa_dir runs/vjepa
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
)

from AE.downstream._shared import (  # noqa: E402
    circular_accuracy, composites, load_inputs, order_by_center,
    sector_mask, wavenumber1_fraction,
)


def metrics_for(emb, emb_dates, anom, anom_dates, stats, rmm, k=6, seed=0):
    lon = stats["lon"]; smask = sector_mask(lon)
    Z = (emb - emb.mean(0)) / (emb.std(0) + 1e-6)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    lab = km.labels_.copy()
    comp = composites(lab, k, emb_dates, anom, anom_dates)
    lab, centers, _ = order_by_center(lab, k, comp, lon, smask)
    comp = composites(lab, k, emb_dates, anom, anom_dates)

    # silhouette (subsampled for speed)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), size=min(3000, len(Z)), replace=False)
    sil = float(silhouette_score(Z[idx], lab[idx]))

    # vs RMM (active days), 8 RMM phases binned to k
    d = rmm.set_index("date")
    common = np.intersect1d(emb_dates, d.index.values)
    e_idx = pd.Index(emb_dates).get_indexer(common)
    amp = d.loc[common, "amplitude"].values
    rphase = (d.loc[common, "phase"].astype(int).values - 1) * k // 8
    active = amp >= 1.0
    c = lab[e_idx][active]; r = rphase[active]

    w1 = float(np.nanmean([wavenumber1_fraction(comp[i]) for i in range(k)]))
    return {
        "silhouette": sil,
        "rmm_ARI": float(adjusted_rand_score(r, c)),
        "rmm_NMI": float(normalized_mutual_info_score(r, c)),
        "rmm_circular_acc": float(circular_accuracy(c, r, k)),
        "wavenumber1_coherence": w1,
        "eastward_monotonic": bool(np.all(np.diff(centers) >= -5.0)),
        "centers_degE": [round(float(x), 1) for x in centers],
        "_labels": lab, "_centers": centers, "_comp": comp,
    }


def run(ae_ckpt: str, jepa_dir: str, k: int = 6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    cfg = torch.load(ae_ckpt, map_location="cpu", weights_only=False)["config"]
    ae_dir, ae_emb, ae_dates, anom, anom_dates, stats, rmm = load_inputs(ae_ckpt, cfg)

    jepa_emb = np.load(os.path.join(jepa_dir, "embeddings.npy"))
    jepa_dates = np.load(os.path.join(jepa_dir, "emb_dates.npy")).astype("datetime64[D]")

    assert ae_emb.shape[1] == jepa_emb.shape[1], (
        f"latent dims differ: AE {ae_emb.shape[1]} vs V-JEPA {jepa_emb.shape[1]} — "
        f"the comparison requires matched dimensions"
    )

    M = {
        "AE": metrics_for(ae_emb, ae_dates, anom, anom_dates, stats, rmm, k=k),
        "V-JEPA": metrics_for(jepa_emb, jepa_dates, anom, anom_dates, stats, rmm, k=k),
    }

    keys = ["silhouette", "rmm_ARI", "rmm_NMI", "rmm_circular_acc",
            "wavenumber1_coherence", "eastward_monotonic", "centers_degE"]
    table = pd.DataFrame({m: {kk: M[m][kk] for kk in keys} for m in M}).T
    csv_path = os.path.join(ae_dir, "ae_vs_jepa.csv")
    table.to_csv(csv_path)

    # comparison bar figure
    bar_keys = ["silhouette", "rmm_ARI", "rmm_NMI", "rmm_circular_acc", "wavenumber1_coherence"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(bar_keys)); w = 0.38
    for i, m in enumerate(["AE", "V-JEPA"]):
        ax.bar(x + (i - 0.5) * w, [M[m][kk] for kk in bar_keys], w, label=m)
    ax.set_xticks(x); ax.set_xticklabels(bar_keys, rotation=20, ha="right")
    ax.set_ylabel("score"); ax.legend()
    ax.set_title(f"AE vs V-JEPA (matched encoder & latent dim={ae_emb.shape[1]}; k={k} phases)")
    fig.tight_layout()
    fpath = os.path.join(ae_dir, "ae_vs_jepa.png")
    fig.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"[ae.compare] matched latent dim = {ae_emb.shape[1]}  (AE & V-JEPA share encoder + dim)")
    print(table.to_string())
    print(f"[ae.compare] table -> {csv_path}\n[ae.compare] figure -> {fpath}")
    return {m: {kk: M[m][kk] for kk in keys} for m in M}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae_ckpt", required=True)
    ap.add_argument("--jepa_dir", default="runs/vjepa")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    run(args.ae_ckpt, args.jepa_dir, k=args.k)


if __name__ == "__main__":
    main()
