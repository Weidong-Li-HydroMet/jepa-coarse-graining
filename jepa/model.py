"""I-JEPA on the shared backbone (arXiv:2301.08243).

From one CONTEXT block, predict the latent representations of ~4 TARGET blocks of
the same map. Targets come from the EMA target encoder applied to the FULL map and
masked at the output (representation) level; the predictor is a narrow ViT taking
context tokens + mask tokens that carry the target positions. L2 loss in latent
space. No augmentation anywhere.

Block sampling adaptations for the 8 x 90 patch grid (lat x lon):
  * target scale (0.15, 0.2) of 720 patches often implies a height > 8 rows; the
    sampler clamps the height to the grid and recomputes the width, so targets can
    span the full latitude band (physically: zonal strips — acceptable in v1).
  * v1 simplification: one context/target geometry is sampled PER BATCH (the paper
    samples per image with batch-shared sizes); revisit at M4 if it matters.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from common.backbone_vit import ViTBackbone, get_3d_sincos_pos_embed
from common.utils import ema_update


class Predictor(nn.Module):
    """Narrow ViT: context tokens + positioned mask tokens -> predicted target latents."""

    def __init__(
        self,
        grid_size: tuple[int, int, int],
        in_dim: int,
        pred_dim: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        L = grid_size[0] * grid_size[1] * grid_size[2]
        self.proj_in = nn.Linear(in_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, L, pred_dim), requires_grad=False)
        self.blocks = nn.ModuleList(
            [Block(pred_dim, num_heads, mlp_ratio, qkv_bias=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(pred_dim)
        self.proj_out = nn.Linear(pred_dim, in_dim)

        pos = get_3d_sincos_pos_embed(pred_dim, grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self, ctx: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor
    ) -> torch.Tensor:
        """ctx: encoded context tokens [B, Nc, D]; ctx/tgt_idx: flat patch indices."""
        B = ctx.shape[0]
        x_ctx = self.proj_in(ctx) + self.pos_embed[:, ctx_idx]
        x_tgt = self.mask_token + self.pos_embed[:, tgt_idx]
        x = torch.cat([x_ctx, x_tgt.expand(B, -1, -1)], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, -tgt_idx.numel():])  # keep mask-token outputs
        return self.proj_out(x)  # [B, Nt, D]


class IJEPA(nn.Module):
    """Context encoder (= shared backbone) + EMA target encoder + predictor + L2 loss."""

    def __init__(
        self,
        backbone: ViTBackbone,
        pred_dim: int = 96,
        pred_depth: int = 4,
        pred_num_heads: int = 4,
        num_targets: int = 4,
        target_scale: tuple[float, float] = (0.15, 0.2),
        target_aspect: tuple[float, float] = (0.75, 1.5),
        context_scale: tuple[float, float] = (0.85, 1.0),
        seed: int | None = None,
    ):
        super().__init__()
        self.context_encoder = backbone
        self.target_encoder = copy.deepcopy(backbone)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(
            backbone.patch_embed.grid_size,
            backbone.embed_dim,
            pred_dim,
            pred_depth,
            pred_num_heads,
        )
        self.num_targets = num_targets
        self.target_scale = target_scale
        self.target_aspect = target_aspect
        self.context_scale = context_scale
        self.rng = np.random.default_rng(seed)

    # ---- block sampling on the patch grid ------------------------------------ #
    def _sample_block(self, scale, aspect) -> np.ndarray:
        """Flat patch indices of one rectangular block on the (gh, gw) grid."""
        gh, gw = self.context_encoder.grid_hw
        s = self.rng.uniform(*scale)
        a = self.rng.uniform(*aspect)
        area = s * gh * gw
        h = min(max(int(round(math.sqrt(area * a))), 1), gh)
        w = min(max(int(round(area / h)), 1), gw)
        top = self.rng.integers(0, gh - h + 1)
        left = self.rng.integers(0, gw - w + 1)
        rows = np.arange(top, top + h)
        cols = np.arange(left, left + w)
        return (rows[:, None] * gw + cols[None, :]).ravel()

    def sample_masks(self, device) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """One geometry per batch: context indices (targets removed) + target index lists."""
        targets = [
            self._sample_block(self.target_scale, self.target_aspect)
            for _ in range(self.num_targets)
        ]
        ctx = self._sample_block(self.context_scale, (1.0, 1.0))
        ctx = np.setdiff1d(ctx, np.concatenate(targets))
        if ctx.size == 0:  # degenerate draw — fall back to everything-but-targets
            L = self.context_encoder.num_patches
            ctx = np.setdiff1d(np.arange(L), np.concatenate(targets))
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).long().to(device)
        return to_t(ctx), [to_t(t) for t in targets]

    # ---- forward -------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] -> scalar L2 loss in latent space."""
        ctx_idx, target_idx = self.sample_masks(x.device)

        with torch.no_grad():
            target_tokens = self.target_encoder(x)  # full map -> [B, L, D]

        tokens = self.context_encoder.embed(x)
        z_ctx = self.context_encoder.encode(tokens[:, ctx_idx])

        loss = 0.0
        for tgt in target_idx:
            pred = self.predictor(z_ctx, ctx_idx, tgt)
            loss = loss + F.mse_loss(pred, target_tokens[:, tgt])
        return loss / len(target_idx)

    @torch.no_grad()
    def momentum_update(self, momentum: float) -> None:
        ema_update(self.context_encoder, self.target_encoder, momentum)

    @property
    def probe_backbone(self) -> ViTBackbone:
        """The encoder the frozen probe should use (target encoder, per the paper)."""
        return self.target_encoder


def build_model(backbone: ViTBackbone, cfg: dict, seed: int | None = None) -> IJEPA:
    """Instantiate from the `method:` section of jepa/config.yaml."""
    return IJEPA(
        backbone,
        pred_dim=cfg["pred_dim"],
        pred_depth=cfg["pred_depth"],
        pred_num_heads=cfg["pred_num_heads"],
        num_targets=cfg["num_targets"],
        target_scale=tuple(cfg["target_scale"]),
        target_aspect=tuple(cfg["target_aspect"]),
        context_scale=tuple(cfg["context_scale"]),
        seed=seed,
    )
