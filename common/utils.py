"""Shared utilities: seeding, config, schedules, checkpoint I/O, logging."""

from __future__ import annotations

import logging
import math
import os
import random
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    """Load a YAML config and return it as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Console (+ optional file) logger; idempotent per name."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def cosine_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    """Cosine interpolation start -> end over total_steps (used for EMA momentum)."""
    if total_steps <= 1:
        return end
    t = min(max(step / (total_steps - 1), 0.0), 1.0)
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def ema_update(student: torch.nn.Module, teacher: torch.nn.Module, momentum: float) -> None:
    """teacher <- momentum * teacher + (1 - momentum) * student (skips frozen params)."""
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        if pt.requires_grad or ps.requires_grad:
            pt.mul_(momentum).add_(ps.detach(), alpha=1.0 - momentum)


def save_checkpoint(path: str, model: torch.nn.Module, backbone: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, epoch: int, cfg: dict) -> None:
    """Save full method state + the bare backbone (what the probe reloads)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "backbone": backbone.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )


def load_backbone_state(path: str, map_location: str = "cpu") -> dict[str, Any]:
    """Return the bare-backbone state_dict from a method checkpoint."""
    return torch.load(path, map_location=map_location, weights_only=False)["backbone"]


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
