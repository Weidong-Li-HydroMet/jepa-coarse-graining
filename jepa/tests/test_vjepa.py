"""Shape / behaviour contracts for the spatiotemporal MJO V-JEPA (synthetic, no cache).

    python -m pytest jepa/tests/test_vjepa.py -q
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from jepa.data.dataset import SyntheticSequence
from jepa.masks.masking import MaskSampler
from jepa.models.encoder import STViTBackbone
from jepa.models.pos_embed import get_1d_sincos_pos_embed_periodic
from jepa.models.vjepa import VJEPA

GRID = (3, 13, 18)
IMG = (13, 144)


def _model():
    enc = STViTBackbone(img_size=IMG, patch_size=(1, 8), num_frames=3, embed_dim=192, depth=2)
    ms = MaskSampler(enc.grid_thw, mode="mixed", seed=0)
    return VJEPA(enc, ms, pred_dim=96, pred_depth=2)


def test_periodic_longitude_is_circular_and_smooth():
    W = 18
    e = get_1d_sincos_pos_embed_periodic(72, W)
    d = np.array([np.linalg.norm(e[0] - e[j]) for j in range(W)])
    assert np.isclose(d[1], d[W - 1])  # seam symmetry
    assert d[1] < d[W // 2]  # neighbour closer than antipode
    assert np.all(np.diff(d[: W // 2 + 1]) > -1e-9)  # monotone to antipode


def test_encoder_token_count():
    enc = STViTBackbone(img_size=IMG, patch_size=(1, 8), num_frames=3, embed_dim=192, depth=2)
    assert enc.grid_thw == GRID
    assert enc.num_patches == 3 * 13 * 18
    x = torch.randn(2, 3, 3, 13, 144)
    assert enc(x).shape == (2, enc.num_patches, 192)


def test_mask_modes_disjoint_and_cover():
    ms = MaskSampler(GRID, sector_width=(3, 6), num_spatial_targets=2, n_future=1)
    for mode in ("spatial", "temporal", "both"):
        ms.mode = mode
        ctx, tgts = ms.sample()
        union = np.concatenate(tgts)
        assert np.intersect1d(ctx, union).size == 0  # disjoint
        assert ctx.size > 0
        assert np.union1d(ctx, union).size == GRID[0] * GRID[1] * GRID[2]  # cover


def test_temporal_mask_hides_last_frame():
    ms = MaskSampler(GRID, mode="temporal", n_future=1)
    _, tgts = ms.sample()
    per = GRID[1] * GRID[2]
    assert np.array_equal(np.sort(tgts[0]), np.arange(2 * per, 3 * per))  # frame T-1


def test_forward_backward_and_no_collapse():
    m = _model()
    x = torch.randn(4, 3, 3, 13, 144)
    loss, st = m(x)
    loss.backward()
    assert torch.isfinite(loss)
    assert st["tgt_std"] > 1e-3  # not collapsed at init
    assert m.predictor.mask_token.grad is not None


def test_embed_clip_pooling():
    m = _model()
    x = torch.randn(5, 3, 3, 13, 144)
    assert m.embed_clip(x, pool="current").shape == (5, 192)
    assert m.embed_clip(x, pool="all").shape == (5, 192)


def test_synthetic_dataset_shape():
    ds = SyntheticSequence(n=8, shape=(3, 3, 13, 144))
    assert ds[0]["x"].shape == (3, 3, 13, 144)
