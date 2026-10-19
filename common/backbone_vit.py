"""Shared ViT backbone for the MJO SSL comparison — the SINGLE SOURCE OF TRUTH.

Copied (encoder only) from the FourierMAE codebase:
    /glade/work/weidong/code/SMAE/SSL-for-2D-Turbulence-b-FourierMAE/src/
        models/vision_transformer.py  (class ViT, encoder part)
        utils/patch_embed.py          (class PatchEmbed)
        utils/pos_embed.py            (3D sin-cos positional embedding)
    (IBM/NASA Prithvi lineage.)

Adaptations, reviewed in 00_problem_setup_and_backbone.ipynb:
  1. project-facing input is [B, C, H, W]; the time dim (T=1) is added internally,
  2. rectangular grids (H != W) supported,
  3. `forward_pooled` returns the mean-pooled token feature for the frozen linear
     probe — this ViT has no CLS token,
  4. `embed` / `encode` are exposed separately so MAE (visible tokens only) and
     I-JEPA (context block only) can run the blocks on a token subset.

MAE / DINO / I-JEPA all import this module. Never redefine the encoder elsewhere —
that would break the fairness of the three-way comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from timm.layers import to_2tuple
from timm.models.vision_transformer import Block


# --------------------------------------------------------------------------- #
# Positional embedding — copied verbatim from utils/pos_embed.py
# --------------------------------------------------------------------------- #
def get_3d_sincos_pos_embed(embed_dim: int, grid_size, cls_token: bool = False):
    """Fixed 3D sin-cos embedding. grid_size = (t, h, w); returns [L, D].

    The embedding dim is split W : H : T = 6 : 6 : 4 sixteenths of D,
    hence the `embed_dim % 16 == 0` requirement.
    """
    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
    """embed_dim: output dim per position; pos: positions (M,). Returns (M, D)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    return np.concatenate([emb_sin, emb_cos], axis=1)


# --------------------------------------------------------------------------- #
# Patch embedding — copied from utils/patch_embed.py (T fixed to 1 for this project)
# --------------------------------------------------------------------------- #
class PatchEmbed(nn.Module):
    """(B, C, T, H, W) -> (B, L, D) via non-overlapping Conv3d."""

    def __init__(
        self,
        img_size=(32, 360),
        patch_size: int = 4,
        num_frames: int = 1,
        tubelet_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 192,
        norm_layer=None,
        flatten: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1]),
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,D,T,H,W -> B,L,D
        return self.norm(x)


# --------------------------------------------------------------------------- #
# The shared encoder
# --------------------------------------------------------------------------- #
class ViTBackbone(nn.Module):
    """FourierMAE ViT encoder with a [B, C, H, W] interface and mean-pool feature."""

    def __init__(
        self,
        img_size=(32, 360),
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        norm_layer=nn.LayerNorm,
        drop_path: float = 0.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size, patch_size, 1, 1, in_chans, embed_dim, norm_layer=None
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_hw = self.patch_embed.grid_size[1:]  # (H/p, W/p)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )

        dpr = drop_path * np.linspace(0.0, 1.0, depth)
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        pos = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=False
        )
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

    # ---- staged forward (used by the SSL heads) ---------------------------- #
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> patch tokens + frozen pos embed, [B, L, D] (pre-blocks)."""
        x = self.patch_embed(x.unsqueeze(2))  # add T=1
        return x + self.pos_embed

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run transformer blocks + final norm on (a subset of) embedded tokens."""
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)

    # ---- standard entry points --------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> tokens [B, L, D]."""
        return self.encode(self.embed(x))

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> mean-pooled feature [B, D] (probe/DINO input; no CLS token)."""
        return self.forward(x).mean(dim=1)


def build_backbone(cfg: dict) -> ViTBackbone:
    """Instantiate the shared backbone from the `backbone:` section of a config."""
    return ViTBackbone(
        img_size=tuple(cfg["img_size"]),
        patch_size=cfg["patch_size"],
        in_chans=cfg["in_chans"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        drop_path=cfg.get("drop_path", 0.0),
        proj_drop=cfg.get("proj_drop", 0.0),
        attn_drop=cfg.get("attn_drop", 0.0),
    )
