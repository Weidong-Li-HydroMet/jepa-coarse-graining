"""Pretrain I-JEPA on daily MJO maps.  Run:  python jepa/train.py --config jepa/config.yaml

The probe checkpoint stores the TARGET encoder (model.probe_backbone).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.backbone_vit import build_backbone
from common.data import build_dataset
from common.utils import (cosine_schedule, get_logger, load_config,
                          save_checkpoint, set_seed)
from jepa.model import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tr = cfg["train"]

    set_seed(tr["seed"])
    device = tr["device"] if torch.cuda.is_available() else "cpu"
    log = get_logger("jepa", os.path.join(tr["out_dir"], "train.log"))
    log.info(f"resolved config: {cfg}")

    train_ds = build_dataset(cfg["data"], "train")
    train_dl = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True,
                          num_workers=tr["num_workers"], pin_memory=True, drop_last=True)

    backbone = build_backbone(cfg["backbone"])
    model = build_model(backbone, cfg["method"], seed=tr["seed"]).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=tr["lr"], weight_decay=tr["weight_decay"])
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-3, total_iters=tr["warmup_epochs"] * len(train_dl))

    m0, m1 = cfg["method"]["ema_momentum"]
    total_steps = tr["epochs"] * len(train_dl)
    step = 0
    for epoch in range(tr["epochs"]):
        model.train()
        t0, tot = time.time(), 0.0
        for batch in train_dl:
            x = batch["x"].to(device, non_blocking=True)
            loss = model(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            warmup.step()
            model.momentum_update(cosine_schedule(m0, m1, step, total_steps))
            step += 1
            tot += loss.item()
        log.info(f"epoch {epoch:03d}  train {tot/len(train_dl):.4f}  "
                 f"({time.time()-t0:.1f}s)")
        save_checkpoint(os.path.join(tr["out_dir"], "ckpt.tar"),
                        model, model.probe_backbone, opt, epoch, cfg)


if __name__ == "__main__":
    main()
