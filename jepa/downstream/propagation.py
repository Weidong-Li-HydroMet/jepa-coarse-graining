"""MJO propagation-type classification from V-JEPA event embeddings.

There are no ground-truth propagation labels, so we follow the task's two-step recipe:

  1. **Derive heuristic labels** from each MJO event's RMM phase-angle trajectory
     (events = contiguous runs of amplitude >= 1, length >= ``min_len`` days):
       * ``standing``  — little net eastward phase advance;
       * ``jumping``   — erratic / non-monotonic trajectory (low forward fraction or a
                         large single-day phase jump);
       * ``slow``      — monotonic eastward propagation, below-median phase speed;
       * ``fast``      — monotonic eastward propagation, above-median phase speed.
     (eastward MJO propagation = counter-clockwise = increasing RMM phase angle.)
  2. **Train a small supervised head** (logistic regression + MLP) on the *event
     embedding* (mean of the daily V-JEPA embeddings over the event) to predict the
     heuristic label, and report held-out accuracy vs the majority-class baseline.  A
     representation that has learned propagation structure should beat the baseline.

    python -m jepa.downstream.propagation --ckpt runs/vjepa/best.tar
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

CLASSES = ["standing", "slow", "fast", "jumping"]


# --------------------------------------------------------------------------- #
def detect_events(rmm: pd.DataFrame, min_len: int = 10) -> list[dict]:
    """Contiguous runs of amplitude >= 1 lasting at least ``min_len`` days."""
    amp = rmm["amplitude"].values
    active = (amp >= 1.0).astype(int)
    edges = np.flatnonzero(np.diff(np.r_[0, active, 0]))
    events = []
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= min_len:
            events.append({"i0": int(a), "i1": int(b)})  # [i0, i1)
    return events


def trajectory_features(rmm: pd.DataFrame, ev: dict) -> dict:
    sl = slice(ev["i0"], ev["i1"])
    r1 = rmm["RMM1"].values[sl]
    r2 = rmm["RMM2"].values[sl]
    theta = np.unwrap(np.arctan2(r2, r1))  # CCW = eastward
    dur = len(theta)
    net_adv_phases = (theta[-1] - theta[0]) / (np.pi / 4)  # in 45° sectors
    dtheta = np.diff(theta)
    mono_frac = float(np.mean(dtheta > 0)) if dur > 1 else 0.0
    max_jump = float(np.max(np.abs(dtheta)) / (np.pi / 4)) if dur > 1 else 0.0
    speed = net_adv_phases / dur  # sectors per day
    return {
        "dur": dur, "net_adv": net_adv_phases, "mono_frac": mono_frac,
        "max_jump": max_jump, "speed": speed, "amp": float(rmm["amplitude"].values[sl].mean()),
    }


def label_events(feats: list[dict]) -> np.ndarray:
    """Rule-based 4-class labels; speed split uses the median of propagating events."""
    net = np.array([f["net_adv"] for f in feats])
    mono = np.array([f["mono_frac"] for f in feats])
    jump = np.array([f["max_jump"] for f in feats])
    speed = np.array([f["speed"] for f in feats])

    standing = np.abs(net) < 1.5
    jumping = (~standing) & ((mono < 0.6) | (jump > 2.0))
    propagating = ~(standing | jumping)
    med = np.median(speed[propagating]) if propagating.any() else 0.0

    labels = np.empty(len(feats), dtype=object)
    labels[standing] = "standing"
    labels[jumping] = "jumping"
    labels[propagating & (speed >= med)] = "fast"
    labels[propagating & (speed < med)] = "slow"
    return labels


def event_embeddings(events, rmm, emb, emb_dates):
    """Mean V-JEPA embedding over each event's days; drop events with no embedding days."""
    emb_idx = pd.Index(emb_dates)
    out, keep = [], []
    for j, ev in enumerate(events):
        dates = rmm["date"].values[ev["i0"]:ev["i1"]]
        rows = emb_idx.get_indexer(dates)
        rows = rows[rows >= 0]
        if rows.size == 0:
            continue
        out.append(emb[rows].mean(0))
        keep.append(j)
    return np.asarray(out, dtype="float32"), np.asarray(keep, dtype=int)


def run(ckpt: str, min_len: int = 10, seed: int = 0):
    import torch
    cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
    out_dir = os.path.dirname(ckpt)
    emb = np.load(os.path.join(out_dir, "embeddings.npy"))
    emb_dates = np.load(os.path.join(out_dir, "emb_dates.npy")).astype("datetime64[D]")
    rmm = pd.read_csv(cfg["data"]["rmm_csv"], parse_dates=["time"])
    rmm["date"] = rmm["time"].values.astype("datetime64[D]")

    events = detect_events(rmm, min_len=min_len)
    feats = [trajectory_features(rmm, ev) for ev in events]
    labels = label_events(feats)

    X, keep = event_embeddings(events, rmm, emb, emb_dates)
    y = labels[keep]
    dist = {c: int((y == c).sum()) for c in CLASSES}
    print(f"[propagation] {len(events)} events (>= {min_len} d), {len(y)} with embeddings")
    print(f"[propagation] heuristic label distribution: {dist}")

    results = {"n_events": len(y), "label_dist": dist}
    baseline = max(dist.values()) / max(1, len(y)) if len(y) else 0.0
    results["majority_baseline"] = float(baseline)

    # supervised head needs enough events and >=2 members per class for a stratified split
    present = [c for c in CLASSES if dist[c] >= 2]
    if len(y) < 12 or len(present) < 2:
        print(f"[propagation] too few events / classes for a supervised head "
              f"(events={len(y)}, classes>=2={present}); reporting label distribution only.")
        np.savez(os.path.join(out_dir, "propagation.npz"),
                 event_i0=np.array([events[j]["i0"] for j in keep]),
                 event_i1=np.array([events[j]["i1"] for j in keep]),
                 labels=y, embeddings=X)
        return results

    # supervised head: heuristic labels from RMM-trajectory  <-  V-JEPA embedding
    Xs = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=seed, stratify=y)
    for name, clf in [
        ("logreg", LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128,), max_iter=1500, random_state=seed)),
    ]:
        clf.fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))
        results[name] = float(acc)
        print(f"[propagation] {name:6s} held-out accuracy = {acc:.3f}  (baseline {baseline:.3f})")
    cm = confusion_matrix(yte, MLPClassifier(hidden_layer_sizes=(128,), max_iter=1500,
                                             random_state=seed).fit(Xtr, ytr).predict(Xte),
                          labels=CLASSES)
    results["confusion_mlp"] = cm.tolist()
    np.savez(os.path.join(out_dir, "propagation.npz"),
             event_i0=np.array([events[j]["i0"] for j in keep]),
             event_i1=np.array([events[j]["i1"] for j in keep]),
             labels=y, embeddings=X)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--min_len", type=int, default=10)
    args = ap.parse_args()
    run(args.ckpt, min_len=args.min_len)


if __name__ == "__main__":
    main()
