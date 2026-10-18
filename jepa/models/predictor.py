"""Narrow-ViT predictor (V-JEPA / I-JEPA style).

Takes the encoded **context** tokens plus learnable **mask tokens** stamped with the
target positions (day, lat, lon via the periodic 3-D positional embedding) and predicts
the target block's latent representations.  Adapted from the 2-D ``jepa/model.py``
predictor, generalized to the spatiotemporal (t,h,w) grid and the periodic longitude
embedding.

Per-batch shared geometry: ``ctx_idx`` / ``tgt_idx`` are 1-D flat patch-index tensors
shared across the batch (the V-JEPA collator shares mask sizes across a batch; here the
whole geometry is shared, the v1 simplification).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from jepa.models.pos_embed import get_3d_sincos_pos_embed_periodic


class Predictor(nn.Module):
    """Context tokens + positioned mask tokens -> predicted target latents."""

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

        pos = get_3d_sincos_pos_embed_periodic(pred_dim, grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self, ctx: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor
    ) -> torch.Tensor:
        """ctx: encoded context tokens ``[B, Nc, D]``; ctx/tgt_idx: flat patch indices."""
        B = ctx.shape[0]
        x_ctx = self.proj_in(ctx) + self.pos_embed[:, ctx_idx]
        x_tgt = self.mask_token + self.pos_embed[:, tgt_idx]
        x = torch.cat([x_ctx, x_tgt.expand(B, -1, -1)], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, -tgt_idx.numel() :])  # keep the mask-token outputs
        return self.proj_out(x)  # [B, Nt, D]
