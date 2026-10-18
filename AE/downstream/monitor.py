"""Decoupled pattern-correlation MJO monitoring index (paper Section 3 / "New Index").

Given the six canonical phase patterns from ``AE.downstream.phases``, diagnose each day's
MJO **location** and **intensity** independently from the observed fields:

  * pattern correlation r_i between the observed (sector-restricted, standardized)
    U850/U200/OLR field and each canonical phase i  (amplitude-insensitive);
  * SoftMax weights  w_i = exp(f·r_i) / Σ exp(f·r_j),  f = 3;
  * hemisphere disambiguation: the dominant phase (max r_i) fixes the active hemisphere;
    weights of opposing-hemisphere phases are zeroed before the weighted averages;
  * location   λ_MJO = Σ w_i λ_i   (λ_i = phase characteristic-centre longitude);
  * intensity  I_MJO = α · sqrt(Σ w_i α_i²),  α = 3,  α_i = projection amplitude
    (regression coefficient of the obs onto phase i) — a weighted RMS on the paper's scale;
  * event = r_max ≥ 0.36, I_MJO ≥ 1, net-eastward, for ≥ 15 consecutive days.

Then quantify, vs the classical RMM, the two failure rates the paper highlights:
spurious propagation (westward / >10°·day⁻¹ jumps) and convective misplacement
(diagnosed centre in a suppressed, positive-OLR-anomaly region).

    python -m AE.downstream.monitor --ckpt runs/ae/best.tar
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from AE.downstream._shared import (  # noqa: E402
    CH, load_inputs, rows_for, sector_mask,
)

F_SMOOTH = 3.0   # SoftMax smoothing factor (paper)
ALPHA = 3.0      # intensity scaling (paper)
R_EVENT = 0.36   # structural-coherence threshold (paper, 25th pct of historical r_max)
I_EVENT = 1.0    # sustained-strength threshold
MIN_EVENT_DAYS = 15
EAST_HEMI = (0, 1)   # P1, P2 (30-120°E)
WEST_HEMI = (4, 5)   # P5, P6 (Pacific)


def _flat_sector(field: np.ndarray, smask: np.ndarray) -> np.ndarray:
    """``(C, H, W)`` -> 1-D vector over the analysis sector (all channels stacked)."""
    return field[:, :, smask].reshape(-1)


def pattern_correlations(obs_vec: np.ndarray, pat_mat: np.ndarray):
    """Pearson r and projection amplitude α of one obs vector onto each phase pattern.

    obs_vec: (M,);  pat_mat: (k, M) demeaned phase patterns.  Returns (r[k], alpha[k]).
    """
    o = obs_vec - obs_vec.mean()
    on = np.linalg.norm(o) + 1e-12
    r = (pat_mat @ o) / (np.linalg.norm(pat_mat, axis=1) * on + 1e-12)
    # projection amplitude (regression coefficient) onto each (raw) pattern
    alpha = (pat_mat @ obs_vec) / ((pat_mat ** 2).sum(1) + 1e-12)
    return r, alpha


def softmax_weights(r: np.ndarray, f: float = F_SMOOTH) -> np.ndarray:
    z = np.exp(f * (r - r.max()))
    return z / z.sum()


def disambiguate(w: np.ndarray, dominant: int) -> np.ndarray:
    """Zero opposing-hemisphere weights based on the dominant phase, then renormalize."""
    w = w.copy()
    if dominant in EAST_HEMI:
        w[list(WEST_HEMI)] = 0.0
    elif dominant in WEST_HEMI:
        w[list(EAST_HEMI)] = 0.0
    s = w.sum()
    return w / s if s > 0 else w


def diagnose(anom, anom_dates, emb_dates, phase_comp, centers, smask):
    """Daily λ_MJO, I_MJO, r_max, dominant phase over all embedding days."""
    pat_mat = np.stack([_flat_sector(phase_comp[i], smask) for i in range(len(centers))])
    pat_mat = pat_mat - pat_mat.mean(1, keepdims=True)  # demean for correlation
    pos = rows_for(emb_dates, anom_dates)
    N, k = len(emb_dates), len(centers)
    lam = np.full(N, np.nan); inten = np.full(N, np.nan)
    rmax = np.full(N, np.nan); dom = np.full(N, -1, int)
    for n in range(N):
        if pos[n] < 0:
            continue
        obs = _flat_sector(np.asarray(anom[pos[n]]), smask)
        r, alpha = pattern_correlations(obs, pat_mat)
        d = int(np.argmax(r))
        w = disambiguate(softmax_weights(r), d)
        lam[n] = float((w * centers).sum())
        inten[n] = float(ALPHA * np.sqrt((w * alpha ** 2).sum()))
        rmax[n] = float(r[d]); dom[n] = d
    return lam, inten, rmax, dom


def rmm_representative_lon(rmm, anom, anom_dates, lon, smask) -> np.ndarray:
    """Empirical representative longitude for each of the 8 RMM phases.

    = longitude of the composite enhanced-convection (min meridional-mean OLR, within the
    sector) over active days assigned to that RMM phase.  Data-driven, so the RMM<->new
    comparison uses the same fields (no arbitrary constants).
    """
    rep = np.full(8, np.nan)
    d = rmm.set_index("date")
    for p in range(1, 9):
        days = d.index[(d["phase"].astype(int) == p) & (d["amplitude"] >= 1.0)].values
        rows = rows_for(np.asarray(days, "datetime64[D]"), anom_dates)
        rows = rows[rows >= 0]
        if rows.size:
            comp = np.asarray(anom[rows, CH["olr"]]).mean(0).mean(0)  # meridional mean (W,)
            j = np.where(smask)[0]
            rep[p - 1] = lon[j[int(np.argmin(comp[j]))]]
    return rep


def rmm_longitude(rmm, rep_lon) -> np.ndarray:
    """Continuous RMM longitude by interpolating the 8 phase anchors over the phase angle."""
    theta = np.arctan2(rmm["RMM2"].values, rmm["RMM1"].values)  # (-pi, pi], CCW eastward
    # RMM phase p occupies a 45° angular sector; anchor angle = its sector centre
    anchor_ang = (np.arange(8) + 0.5) * (np.pi / 4) - np.pi  # centres of phases 1..8
    order = np.argsort(anchor_ang)
    a, L = anchor_ang[order], np.unwrap(rep_lon[order] * np.pi / 180.0)  # unwrap to avoid seam jumps
    lam = np.interp(theta, a, L, period=2 * np.pi) * 180.0 / np.pi
    return lam


def daily_speed(lam: np.ndarray) -> np.ndarray:
    """Centred-difference zonal phase speed (°·day⁻¹); NaN at ends/gaps."""
    c = np.full_like(lam, np.nan)
    c[1:-1] = (lam[2:] - lam[:-2]) / 2.0
    return c


def misplacement_rate(lam, days, anom, anom_dates, lon, smask) -> float:
    """Fraction of days whose diagnosed centre sits in a suppressed (positive-OLR) region."""
    pos = rows_for(days, anom_dates)
    j = np.where(smask)[0]
    bad = tot = 0
    for n in range(len(days)):
        if pos[n] < 0 or not np.isfinite(lam[n]):
            continue
        prof = np.asarray(anom[pos[n], CH["olr"]]).mean(0)  # standardized meridional-mean OLR
        jj = j[int(np.argmin(np.abs(lon[j] - lam[n])))]     # nearest sector longitude
        tot += 1
        if prof[jj] > 0:   # positive OLR anomaly = suppressed convection
            bad += 1
    return bad / max(1, tot)


def spurious_rate(c: np.ndarray) -> float:
    """Fraction of valid days with westward or >10°·day⁻¹ propagation."""
    v = c[np.isfinite(c)]
    if v.size == 0:
        return 0.0
    return float(np.mean((v < 0) | (np.abs(v) > 10.0)))


def find_events(lam, inten, rmax, dom):
    """Consecutive runs (>=15 d) meeting structure/strength/eastward criteria."""
    ok = (rmax >= R_EVENT) & (inten >= I_EVENT) & np.isfinite(lam)
    events, i, N = [], 0, len(ok)
    while i < N:
        if not ok[i]:
            i += 1; continue
        j = i
        while j < N and ok[j]:
            j += 1
        seg = slice(i, j)
        if (j - i) >= MIN_EVENT_DAYS:
            net = lam[j - 1] - lam[i]
            if net > 0:  # net eastward
                events.append((i, j, float(net), float(inten[seg].mean())))
        i = j
    return events


def run(ckpt: str):
    import torch
    cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
    out_dir, emb, emb_dates, anom, anom_dates, stats, rmm = load_inputs(ckpt, cfg)
    lon = stats["lon"]; smask = sector_mask(lon)

    ph = np.load(os.path.join(out_dir, "ae_phase_labels.npz"))
    phase_comp = ph["composites_som"]      # (6, C, H, W) standardized
    centers = ph["som_centers"]            # (6,) °E

    lam, inten, rmax, dom = diagnose(anom, anom_dates, emb_dates, phase_comp, centers, smask)

    # classical RMM longitude on the same days
    rep = rmm_representative_lon(rmm, anom, anom_dates, lon, smask)
    rmm_full = rmm.copy()
    rmm_full["lam"] = rmm_longitude(rmm_full, rep)
    rmm_by_date = rmm_full.set_index("date")
    common = np.intersect1d(emb_dates, rmm_by_date.index.values)
    e_idx = pd.Index(emb_dates).get_indexer(common)
    amp = rmm_by_date.loc[common, "amplitude"].values
    rmm_lam_c = rmm_by_date.loc[common, "lam"].values
    new_lam_c = lam[e_idx]
    active = amp >= 1.0

    # propagation + misplacement on active days
    new_c = daily_speed(new_lam_c); rmm_c = daily_speed(rmm_lam_c)
    res = {
        "n_active_days": int(active.sum()),
        "new_spurious_propagation": spurious_rate(new_c[active]),
        "rmm_spurious_propagation": spurious_rate(rmm_c[active]),
        "new_misplacement": misplacement_rate(new_lam_c[active], common[active], anom, anom_dates, lon, smask),
        "rmm_misplacement": misplacement_rate(rmm_lam_c[active], common[active], anom, anom_dates, lon, smask),
    }
    events = find_events(lam, inten, rmax, dom)
    res["n_events"] = len(events)
    res["r_event_threshold"] = R_EVENT
    # intensity classification of events
    cls = {"weak": 0, "medium": 0, "strong": 0, "extreme": 0}
    for *_, mi in events:
        cls["weak" if mi < 1.5 else "medium" if mi < 2.0 else "strong" if mi < 2.5 else "extreme"] += 1
    res["event_intensity_class"] = cls

    np.savez(os.path.join(out_dir, "ae_monitor.npz"),
             dates=emb_dates, lam=lam, intensity=inten, rmax=rmax, dom=dom,
             rmm_rep_lon=rep)

    print(f"[ae.monitor] decoupled index over {len(emb_dates)} days, {int(active.sum())} active")
    print(f"[ae.monitor] spurious propagation:  new {res['new_spurious_propagation']*100:.2f}%  "
          f"vs RMM {res['rmm_spurious_propagation']*100:.2f}%")
    print(f"[ae.monitor] convective misplacement: new {res['new_misplacement']*100:.2f}%  "
          f"vs RMM {res['rmm_misplacement']*100:.2f}%")
    print(f"[ae.monitor] MJO events (>= {MIN_EVENT_DAYS} d, r>={R_EVENT}, I>=1, eastward): "
          f"{len(events)}  intensity {cls}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    run(args.ckpt)


if __name__ == "__main__":
    main()
