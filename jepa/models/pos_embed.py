"""3-D sin-cos positional embedding with a **periodic longitude** axis.

The tropical maps are a full 0-360° longitude circle, so the longitude positional
embedding must wrap: patch column ``W`` should be adjacent to column ``0``.  The
standard transformer 1-D sin-cos embedding (used in ``common.backbone_vit``) is *not*
periodic — its frequencies are ``1/10000**(2i/d)`` and position ``0`` is far from
position ``W-1`` in embedding space.

Here the longitude axis uses integer harmonics of the fundamental ``2π/W``:

    e_lon(j) = [sin(k·2π j/W), cos(k·2π j/W)]  for k = 1 .. d/2,

which is exactly periodic with period ``W``.  Latitude and lead-time (day) axes are
bounded and non-cyclic, so they keep the standard sin-cos embedding from
``common.backbone_vit`` (single source of truth).  The W:H:T = 6:6:4 dimension split
and the (t,h,w) tiling order match ``get_3d_sincos_pos_embed`` so this is a drop-in,
periodicity-aware replacement.
"""

from __future__ import annotations

import numpy as np

from common.backbone_vit import get_1d_sincos_pos_embed_from_grid


def get_1d_sincos_pos_embed_periodic(
    embed_dim: int, n: int, decay: float = 0.6
) -> np.ndarray:
    """Smooth low-pass periodic 1-D embedding for ``n`` circle positions -> (n, embed_dim).

    Uses **integer** harmonics of the fundamental ``2π/n`` so the code is exactly periodic
    (position ``n`` coincides with position ``0``), capped below the Nyquist limit
    ``(n-1)//2``.  Harmonic amplitudes decay geometrically (``decay**(k-1)``), so the
    fundamental ``(sin λ, cos λ)`` dominates and the kernel is low-pass: nearby longitudes
    — including across the 0/360° seam — get similar embeddings, the antipode is farthest.
    The harmonic block is zero-padded to ``embed_dim`` (equal-amplitude tiling would
    flatten the kernel into a non-local delta).
    """
    assert embed_dim % 2 == 0, "periodic embed dim must be even"
    half = embed_dim // 2
    k_max = max(1, min(half, (n - 1) // 2))  # below Nyquist
    k = np.arange(1, k_max + 1, dtype=np.float64)
    amp = decay ** (k - 1)  # low-pass: fundamental dominates
    theta = 2.0 * np.pi * np.arange(n, dtype=np.float64) / n  # (n,)
    out = np.einsum("m,d->md", theta, k)  # (n, k_max)
    base = np.concatenate([amp * np.sin(out), amp * np.cos(out)], axis=1)  # (n, 2*k_max)
    emb = np.zeros((n, embed_dim))
    emb[:, : base.shape[1]] = base  # zero-pad to embed_dim
    return emb


def get_3d_sincos_pos_embed_periodic(embed_dim: int, grid_size, cls_token: bool = False):
    """Fixed 3-D sin-cos embedding with periodic longitude. grid = (t, h, w) -> (L, D).

    Dimension split W:H:T = 6:6:4 sixteenths of D (so ``embed_dim % 16 == 0``); the W
    (longitude) block is periodic, the H (latitude) and T (day) blocks are standard.
    """
    assert embed_dim % 16 == 0
    t_size, h_size, w_size = grid_size

    w_dim = embed_dim // 16 * 6
    h_dim = embed_dim // 16 * 6
    t_dim = embed_dim // 16 * 4

    w_pos = get_1d_sincos_pos_embed_periodic(w_dim, w_size)  # periodic longitude
    h_pos = get_1d_sincos_pos_embed_from_grid(h_dim, np.arange(h_size))  # latitude
    t_pos = get_1d_sincos_pos_embed_from_grid(t_dim, np.arange(t_size))  # lead-time/day

    w_pos = np.tile(w_pos, (t_size * h_size, 1))
    h_pos = np.tile(np.repeat(h_pos, w_size, axis=0), (t_size, 1))
    t_pos = np.repeat(t_pos, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos, h_pos, t_pos), axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed
