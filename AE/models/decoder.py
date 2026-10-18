"""Lightweight spatial decoder: latent vector -> full target-day field.

The paper's PhysAnchor-MJO-AE decoder "mirrors" the encoder and maps the encoder output
back to the full spatial domain to reconstruct day ``t``.  Here the encoder is the shared
V-JEPA ``STViTBackbone`` and the per-clip latent is a single vector ``(B, D)`` (D = the
matched V-JEPA ``embed_dim``).  This decoder turns that one vector back into the 2-D field
of the target day ``(B, C, H, W)``:

  1. broadcast the latent to one query token per output spatial patch and add a 2-D
     positional embedding (periodic longitude, reusing ``jepa.models.pos_embed`` — the
     same source of truth as the encoder);
  2. run a few transformer blocks (MAE-decoder style);
  3. linear-project each patch token to its pixels and un-patchify to ``(B, C, H, W)``.

It is deliberately small and swappable — the encoder is what must stay identical to
V-JEPA; the decoder only exists to provide the reconstruction objective.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from jepa.models.pos_embed import get_3d_sincos_pos_embed_periodic


def unpatchify(x: torch.Tensor, gh: int, gw: int, ph: int, pw: int, C: int) -> torch.Tensor:
    """``(B, gh*gw, C*ph*pw)`` patch tokens -> ``(B, C, gh*ph, gw*pw)`` field.

    Patch order is row-major over (lat, lon) — i.e. flat index ``p = h*gw + w`` — matching
    ``common.backbone_vit.PatchEmbed`` (Conv3d output flattened in t,h,w order, T=1 here).
    """
    B = x.shape[0]
    x = x.reshape(B, gh, gw, C, ph, pw)
    x = x.permute(0, 3, 1, 4, 2, 5)  # B, C, gh, ph, gw, pw
    return x.reshape(B, C, gh * ph, gw * pw)


class SpatialDecoder(nn.Module):
    """Per-clip latent ``(B, D)`` -> target-day field ``(B, C, H, W)``."""

    def __init__(
        self,
        latent_dim: int,
        grid_hw: tuple[int, int],   # (H/ph, W/pw) patch grid of one day
        patch_hw: tuple[int, int],  # (ph, pw)
        out_chans: int,
        dec_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        gh, gw = grid_hw
        ph, pw = patch_hw
        assert dec_dim % 16 == 0, "dec_dim must be % 16 == 0 for the periodic pos embed"
        self.gh, self.gw, self.ph, self.pw, self.C = gh, gw, ph, pw, out_chans
        self.n_patches = gh * gw

        self.proj_in = nn.Linear(latent_dim, dec_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, dec_dim), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [Block(dec_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
             for _ in range(depth)]
        )
        self.norm = norm_layer(dec_dim)
        self.head = nn.Linear(dec_dim, ph * pw * out_chans)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        # 2-D spatial pos embed = the (1, gh, gw) slice of the periodic 3-D embedding.
        pos = get_3d_sincos_pos_embed_periodic(
            self.pos_embed.shape[-1], (1, self.gh, self.gw), cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: ``(B, latent_dim)`` -> reconstructed field ``(B, C, H, W)``."""
        x = self.proj_in(z)[:, None, :] + self.pos_embed  # (B, n_patches, dec_dim)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)  # (B, n_patches, C*ph*pw)
        return unpatchify(x, self.gh, self.gw, self.ph, self.pw, self.C)


def build_decoder(cfg_enc: dict, cfg_dec: dict, latent_dim: int) -> SpatialDecoder:
    """Instantiate from the ``encoder:`` (for grid/patch) and ``decoder:`` config blocks."""
    H, W = tuple(cfg_enc["img_size"])
    ph, pw = tuple(cfg_enc["patch_size"])
    return SpatialDecoder(
        latent_dim=latent_dim,
        grid_hw=(H // ph, W // pw),
        patch_hw=(ph, pw),
        out_chans=cfg_enc["in_chans"],
        dec_dim=cfg_dec.get("dec_dim", 128),
        depth=cfg_dec.get("depth", 4),
        num_heads=cfg_dec.get("num_heads", 4),
        mlp_ratio=cfg_dec.get("mlp_ratio", 4.0),
    )
