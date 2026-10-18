"""Single-GPU pretraining for the spatiotemporal MJO V-JEPA.

    python jepa/train_vjepa.py --config jepa/config_vjepa.yaml            # full run (1 GPU)
    python jepa/train_vjepa.py --config jepa/config_vjepa.yaml --smoke    # CPU smoke test
    python jepa/train_vjepa.py --config jepa/config_vjepa.yaml --resume runs/vjepa/last.tar

The checkpoint stores the full model plus the bare EMA target encoder (the feature used
downstream).  AMP, cosine LR + warmup, EMA momentum schedule, grad clipping, and a
collapse monitor (target-embedding std) are all on by default.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import cosine_schedule, get_logger, set_seed  # noqa: E402
from jepa.data.dataset import SyntheticSequence, build_dataset  # noqa: E402
from jepa.masks.masking import build_mask_sampler  # noqa: E402
from jepa.models.encoder import build_encoder  # noqa: E402
from jepa.models.vjepa import build_vjepa  # noqa: E402


def lr_at(step: int, warmup_steps: int, total_steps: int, lr: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def evaluate(model, dl, device, amp: bool) -> dict:
    model.eval()
    agg = {"loss": 0.0, "tgt_std": 0.0, "ctx_std": 0.0}
    n = 0
    for batch in dl:
        x = batch["x"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            _, stats = model(x)
        for k in agg:
            agg[k] += stats[k]
        n += 1
    return {k: v / max(1, n) for k, v in agg.items()}


def build_loaders(cfg, smoke: bool):
    tr = cfg["train"]
    if smoke:
        ds = SyntheticSequence(n=256, shape=(cfg["encoder"]["in_chans"],
                                             cfg["encoder"]["num_frames"],
                                             *cfg["encoder"]["img_size"]))
        vds = SyntheticSequence(n=64, shape=(cfg["encoder"]["in_chans"],
                                             cfg["encoder"]["num_frames"],
                                             *cfg["encoder"]["img_size"]), seed=1)
    else:
        seq = cfg["data"]["seq_len"]
        ds = build_dataset(cfg["data"], "train", seq_len=seq)
        vds = build_dataset(cfg["data"], "val", seq_len=seq)
    nw = 0 if smoke else tr["num_workers"]
    bs = 16 if smoke else tr["batch_size"]
    train_dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw,
                          pin_memory=not smoke, drop_last=True)
    val_dl = DataLoader(vds, batch_size=bs, shuffle=False, num_workers=nw,
                        pin_memory=not smoke, drop_last=False)
    return train_dl, val_dl


def save_ckpt(path, model, opt, scaler, epoch, step, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch, "step": step,
        "model": model.state_dict(),
        "probe_encoder": model.probe_encoder.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "config": cfg,
    }, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic CPU run")
    ap.add_argument("--smoke-steps", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--resume", default='/glade/campaign/univ/uchi0018/weidong/SSL_project/Self_supervised_learning/MJO/runs/vjepa/last.tar')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tr = cfg["train"]
    if args.epochs is not None:
        tr["epochs"] = args.epochs
    set_seed(tr["seed"])

    device = "cuda" if (tr["device"] == "cuda" and torch.cuda.is_available() and not args.smoke) else "cpu"
    amp = bool(tr.get("amp", True)) and device == "cuda"
    out_dir = tr["out_dir"]
    log = get_logger("vjepa", os.path.join(out_dir, "train.log"))

    train_dl, val_dl = build_loaders(cfg, args.smoke)

    encoder = build_encoder(cfg["encoder"])
    mask_sampler = build_mask_sampler(cfg["mask"], encoder.grid_thw, seed=tr["seed"])
    model = build_vjepa(encoder, mask_sampler, cfg["method"]).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"grid (T,H,W)={encoder.grid_thw}  tokens={encoder.num_patches}  "
             f"trainable params={n_par/1e6:.2f}M  device={device}  amp={amp}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=tr["lr"], weight_decay=tr["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    epochs = 2 if args.smoke else tr["epochs"]
    steps_per_epoch = len(train_dl)
    print(steps_per_epoch)
    total_steps = epochs * steps_per_epoch
    warmup_steps = (1 if args.smoke else tr["warmup_epochs"]) * steps_per_epoch
    m0, m1 = cfg["method"]["ema_momentum"]

    start_epoch, step, best = 0, 0, float("inf")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        log.info(f"resumed from {args.resume} at epoch {start_epoch}")

    smoke_losses = []
    for epoch in range(start_epoch, epochs):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for batch in train_dl:
            x = batch["x"].to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, warmup_steps, total_steps, tr["lr"], tr["lr_min"])
            with torch.amp.autocast("cuda", enabled=amp):
                loss, stats = model(x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tr.get("grad_clip", 0):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            scaler.step(opt)
            scaler.update()
            model.momentum_update(cosine_schedule(m0, m1, step, total_steps))
            step += 1
            tot += stats["loss"]
            seen += x.size(0)
            smoke_losses.append(stats["loss"])
            if step % tr.get("log_every", 50) == 0:
                ips = seen / (time.time() - t0)
                log.info(f"e{epoch:03d} s{step:06d} loss {stats['loss']:.4f} "
                         f"pred {stats['pred_loss']:.4f} tgt_std {stats['tgt_std']:.3f} "
                         f"ctx_std {stats['ctx_std']:.3f} lr {opt.param_groups[0]['lr']:.2e} "
                         f"{ips:.0f} clips/s")
            if args.smoke and step >= args.smoke_steps:
                break
        if args.smoke and step >= args.smoke_steps:
            break

        if (epoch % tr.get("eval_every", 1) == 0) or epoch == epochs - 1:
            val = evaluate(model, val_dl, device, amp)
            log.info(f"[epoch {epoch:03d}] train {tot/max(1,steps_per_epoch):.4f}  "
                     f"val {val['loss']:.4f}  val_tgt_std {val['tgt_std']:.3f}  "
                     f"({time.time()-t0:.1f}s)")
            save_ckpt(os.path.join(out_dir, "last.tar"), model, opt, scaler, epoch, step, cfg)
            if val["loss"] < best:
                best = val["loss"]
                save_ckpt(os.path.join(out_dir, "best.tar"), model, opt, scaler, epoch, step, cfg)

    if args.smoke:
        first = sum(smoke_losses[:5]) / max(1, len(smoke_losses[:5]))
        last = sum(smoke_losses[-5:]) / max(1, len(smoke_losses[-5:]))
        tgt_std = stats["tgt_std"]
        log.info(f"[smoke] steps={len(smoke_losses)} loss {first:.4f} -> {last:.4f} "
                 f"(decreased={last < first})  final tgt_std={tgt_std:.3f} "
                 f"(collapse if ~0)")
        save_ckpt(os.path.join(out_dir, "smoke.tar"), model, opt, scaler, 0, step, cfg)
        ok = (last < first) and (tgt_std > 1e-2)
        log.info(f"[smoke] PASS={ok}")


if __name__ == "__main__":
    main()
