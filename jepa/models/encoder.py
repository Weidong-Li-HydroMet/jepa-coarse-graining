"""Spatiotemporal ViT encoder for 3-day MJO clips (V-JEPA context/target encoder).

Reuses ``common.backbone_vit.PatchEmbed`` (Conv3d patching with a temporal axis) and
``timm`` transformer blocks — the same encoder family as the 2-D comparison backbone,
so nothing is reinvented.  Differences from ``common.ViTBackbone``:

  * input is a clip ``(B, C, T, H, W)`` (T frames = T days), patched with
    ``tubelet_size=1`` so each day keeps its own temporal token (needed for the
    future-day prediction task);
  * a rectangular spatial patch ``(p_lat, p_lon)`` (latitude is only 13 rows wide);
  * a **periodic** longitude positional embedding (``get_3d_sincos_pos_embed_periodic``).

``embed`` / ``encode`` are exposed separately so the context encoder can run the
transformer on a *subset* of tokens (the visible/context block), exactly as I-JEPA /
V-JEPA do.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from common.backbone_vit import PatchEmbed
from jepa.models.pos_embed import get_3d_sincos_pos_embed_periodic


class STViTBackbone(nn.Module):
    """Spatiotemporal ViT encoder with a ``(B, C, T, H, W)`` interface."""

    def __init__(
        self,
        img_size=(13, 144),
        patch_size=(1, 8),
        num_frames: int = 3,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        norm_layer=nn.LayerNorm,
        drop_path: float = 0.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        ph, pw = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        H, W = img_size
        assert H % ph == 0 and W % pw == 0, (
            f"patch {(ph, pw)} must divide grid {(H, W)} exactly "
            f"(latitude has {H} rows; use p_lat=1 if it is not divisible)"
        )
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=(ph, pw),
            num_frames=num_frames,
            tubelet_size=1,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=None,
        )
        self.grid_thw = self.patch_embed.grid_size  # (T, H/ph, W/pw)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )
        dpr = drop_path * np.linspace(0.0, 1.0, depth)
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                    proj_drop=proj_drop, attn_drop=attn_drop, drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        pos = get_3d_sincos_pos_embed_periodic(self.embed_dim, self.grid_thw, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ---- staged forward ---------------------------------------------------- #
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, H, W)`` -> patch tokens + periodic pos embed ``(B, L, D)``."""
        return self.patch_embed(x) + self.pos_embed

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run transformer blocks + final norm on (a subset of) embedded tokens."""
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, H, W)`` -> all tokens ``(B, L, D)``."""
        return self.encode(self.embed(x))

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pooled clip feature ``(B, D)`` (no CLS token in this ViT)."""
        return self.forward(x).mean(dim=1)

    # ---- grid bookkeeping (used by masks + downstream pooling) ------------- #
    def frame_token_index(self, t: int) -> np.ndarray:
        """Flat token indices belonging to frame ``t`` (0=oldest .. T-1=current day)."""
        T, gh, gw = self.grid_thw
        per = gh * gw
        return np.arange(t * per, (t + 1) * per)


def build_encoder(cfg: dict) -> STViTBackbone:
    """Instantiate from the ``encoder:`` section of the config."""
    return STViTBackbone(
        img_size=tuple(cfg["img_size"]),
        patch_size=tuple(cfg["patch_size"]),
        num_frames=cfg["num_frames"],
        in_chans=cfg["in_chans"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        drop_path=cfg.get("drop_path", 0.0),
        proj_drop=cfg.get("proj_drop", 0.0),
        attn_drop=cfg.get("attn_drop", 0.0),
    )
