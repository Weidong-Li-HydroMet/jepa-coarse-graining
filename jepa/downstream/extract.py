"""Extract one V-JEPA embedding per day from a trained checkpoint.

For every anchor day ``t`` with two days of history, run the EMA *target* encoder over
the clip ``[t-2, t-1, t]`` and mean-pool the **current-day** tokens (the day-t frame) to
get a single vector representing day ``t`` in the context of its recent evolution.

    python -m jepa.downstream.extract --ckpt runs/vjepa/best.tar
    # -> runs/vjepa/embeddings.npy [N, D] + runs/vjepa/emb_dates.npy [N]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from jepa.data.dataset import MJOSequence  # noqa: E402
from jepa.models.encoder import build_encoder  # noqa: E402


@torch.no_grad()
def extract(ckpt_path: str, pool: str = "current", device: str | None = None,
            out_dir: str | None = None, batch_size: int = 256):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    enc = build_encoder(cfg["encoder"]).to(device)
    enc.load_state_dict(ck["probe_encoder"])
    enc.eval()

    d = cfg["data"]
    years = range(d["year_start"], d["year_end"] + 1)
    ds = MJOSequence(d["cache_dir"], list(years), seq_len=d["seq_len"])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    cur_idx = torch.from_numpy(enc.frame_token_index(enc.num_frames - 1)).long().to(device)
    embs = []
    for batch in dl:
        x = batch["x"].to(device, non_blocking=True)
        tokens = enc(x)  # [B, L, D]
        emb = tokens.mean(1) if pool == "all" else tokens[:, cur_idx].mean(1)
        embs.append(emb.cpu().numpy())
    embeddings = np.concatenate(embs, 0).astype("float32")
    dates = ds.anchor_dates().astype("datetime64[D]")

    out_dir = out_dir or os.path.dirname(ckpt_path)
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(out_dir, "emb_dates.npy"), dates)
    print(f"[extract] {embeddings.shape} embeddings ({pool}-day pool) "
          f"{dates[0]} -> {dates[-1]} -> {out_dir}")
    return embeddings, dates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pool", default="current", choices=["current", "all"])
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    extract(args.ckpt, pool=args.pool, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
