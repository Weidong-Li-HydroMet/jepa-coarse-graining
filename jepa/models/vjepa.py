"""Spatiotemporal V-JEPA: context encoder + EMA target encoder + predictor + latent loss.

From the visible (context) tokens of a 3-day clip, predict — in latent space — the
representations of the masked target blocks (a hidden longitude sector, or the future
day).  Targets come from the EMA *target* encoder applied to the full clip and selected
at the representation level (I-JEPA recipe); the loss is smooth-L1 / L2 between the
predictor output and the (stop-grad) target latents.  No pixel reconstruction.

Anti-collapse: the EMA-target + narrow-predictor asymmetry (the I-JEPA mechanism), plus
a logged embedding-variance monitor and an *optional* VICReg-style variance hinge on the
online context embeddings (``var_coeff`` > 0) that can be switched on if collapse appears.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils import ema_update
from jepa.masks.masking import MaskSampler
from jepa.models.encoder import STViTBackbone
from jepa.models.predictor import Predictor


class VJEPA(nn.Module):
    def __init__(
        self,
        encoder: STViTBackbone,
        mask_sampler: MaskSampler,
        pred_dim: int = 96,
        pred_depth: int = 4,
        pred_num_heads: int = 4,
        loss_type: str = "smooth_l1",  # 'smooth_l1' | 'mse'
        var_coeff: float = 0.0,  # VICReg variance hinge on online embeddings
    ):
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(
            encoder.grid_thw, encoder.embed_dim, pred_dim, pred_depth, pred_num_heads
        )
        self.mask_sampler = mask_sampler
        self.loss_type = loss_type
        self.var_coeff = float(var_coeff)

    # ---- loss helpers ------------------------------------------------------- #
    def _pred_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        return F.smooth_l1_loss(pred, target)

    @staticmethod
    def _variance_hinge(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """VICReg variance term: encourage per-feature std >= 1 (anti-collapse)."""
        z = z.reshape(-1, z.shape[-1])
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))

    # ---- forward ------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """x: ``(B, C, T, H, W)`` -> (scalar loss, diagnostics dict)."""
        ctx_np, targets_np = self.mask_sampler.sample()
        dev = x.device
        ctx_idx = torch.from_numpy(ctx_np).long().to(dev)
        target_idx = [torch.from_numpy(t).long().to(dev) for t in targets_np]

        with torch.no_grad():
            target_tokens = self.target_encoder(x)  # [B, L, D]

        tokens = self.context_encoder.embed(x)
        z_ctx = self.context_encoder.encode(tokens[:, ctx_idx])

        pred_loss = x.new_zeros(())
        for tgt in target_idx:
            pred = self.predictor(z_ctx, ctx_idx, tgt)
            pred_loss = pred_loss + self._pred_loss(pred, target_tokens[:, tgt])
        pred_loss = pred_loss / len(target_idx)

        loss = pred_loss
        var_term = self._variance_hinge(z_ctx) if self.var_coeff > 0 else z_ctx.new_zeros(())
        if self.var_coeff > 0:
            loss = loss + self.var_coeff * var_term

        with torch.no_grad():
            stats = {
                "loss": float(loss.detach()),
                "pred_loss": float(pred_loss.detach()),
                # collapse monitor: std of target latents over batch+tokens, per-feature mean
                "tgt_std": float(target_tokens.reshape(-1, target_tokens.shape[-1]).std(0).mean()),
                "ctx_std": float(z_ctx.reshape(-1, z_ctx.shape[-1]).std(0).mean()),
                "n_ctx": int(ctx_idx.numel()),
                "n_tgt": int(sum(t.numel() for t in target_idx)),
                "var_term": float(var_term.detach()),
            }
        return loss, stats

    @torch.no_grad()
    def momentum_update(self, momentum: float) -> None:
        ema_update(self.context_encoder, self.target_encoder, momentum)

    # ---- downstream embedding (uses the EMA target encoder, per I-JEPA) ------ #
    @property
    def probe_encoder(self) -> STViTBackbone:
        return self.target_encoder

    @torch.no_grad()
    def embed_clip(self, x: torch.Tensor, pool: str = "current") -> torch.Tensor:
        """Per-clip embedding ``(B, D)``. pool='current' (last-day tokens) | 'all'."""
        enc = self.target_encoder
        tokens = enc(x)  # [B, L, D]
        if pool == "all":
            return tokens.mean(dim=1)
        idx = torch.from_numpy(enc.frame_token_index(enc.num_frames - 1)).long().to(x.device)
        return tokens[:, idx].mean(dim=1)


def build_vjepa(encoder, mask_sampler, cfg_method: dict) -> VJEPA:
    """Instantiate from the ``method:`` config section."""
    return VJEPA(
        encoder,
        mask_sampler,
        pred_dim=cfg_method["pred_dim"],
        pred_depth=cfg_method["pred_depth"],
        pred_num_heads=cfg_method["pred_num_heads"],
        loss_type=cfg_method.get("loss_type", "smooth_l1"),
        var_coeff=cfg_method.get("var_coeff", 0.0),
    )
