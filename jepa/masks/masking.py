"""Masking for the MJO V-JEPA — two physically-motivated families, combinable.

Grid is ``(T, GH, GW)`` patches (day, latitude, longitude); flat index of patch
``(t, h, w)`` is ``t*GH*GW + h*GW + w``.

**Spatial sector (tube) mask** — a contiguous block of longitude patch-columns,
**wrapping** around 0/360°, masked across *all* latitudes and *all* days (a "tube",
as in V-JEPA ``random_tube``).  Predicting a hidden longitude sector from the visible
ones is what forces the translation-tolerant zonal structure the MJO needs.

**Temporal future mask** — hide the last ``n_future`` day(s) entirely and predict
their embeddings from the earlier days.  This forces the encoder + predictor to learn
the eastward propagation / dynamics.

Both keep context and target tokens disjoint (I-JEPA convention).  ``mode``:
  * ``spatial``  — one or more longitude-sector tubes are the targets;
  * ``temporal`` — the future day(s) are the target;
  * ``mixed``    — pick ``spatial`` or ``temporal`` at random per call (default);
  * ``both``     — mask a sector *and* the future day in the same sample.

``sample()`` returns ``(ctx_idx, [target_idx, ...])`` as numpy int arrays of flat patch
indices; the model converts them to device tensors.
"""

from __future__ import annotations

import numpy as np


class MaskSampler:
    def __init__(
        self,
        grid_thw: tuple[int, int, int],
        mode: str = "mixed",
        sector_width=(3, 6),  # range of masked longitude-patch columns
        num_spatial_targets: int = 2,
        n_future: int = 1,  # number of trailing days masked in the temporal task
        seed: int | None = None,
    ):
        self.T, self.GH, self.GW = grid_thw
        self.per_frame = self.GH * self.GW
        self.mode = mode
        self.sector_width = tuple(sector_width)
        self.num_spatial_targets = int(num_spatial_targets)
        self.n_future = int(n_future)
        assert 1 <= self.n_future < self.T, "n_future must leave >=1 context frame"
        self.rng = np.random.default_rng(seed)

    # ---- primitive blocks --------------------------------------------------- #
    def _sector_tube(self) -> np.ndarray:
        """Flat indices of one wrap-around longitude sector across all lats and days."""
        sw = int(self.rng.integers(self.sector_width[0], self.sector_width[1] + 1))
        sw = min(sw, self.GW)
        start = int(self.rng.integers(0, self.GW))
        cols = (start + np.arange(sw)) % self.GW  # wraps around 0/360°
        t = np.arange(self.T)[:, None, None]
        h = np.arange(self.GH)[None, :, None]
        idx = t * self.per_frame + h * self.GW + cols[None, None, :]
        return idx.ravel()

    def _future_frames(self) -> np.ndarray:
        """Flat indices of the last ``n_future`` frames (all lats/lons)."""
        f0 = self.T - self.n_future
        return np.arange(f0 * self.per_frame, self.T * self.per_frame)

    # ---- geometry ----------------------------------------------------------- #
    def _resolve_mode(self) -> str:
        if self.mode == "mixed":
            return self.rng.choice(["spatial", "temporal"])
        return self.mode

    def sample(self) -> tuple[np.ndarray, list[np.ndarray]]:
        mode = self._resolve_mode()
        targets: list[np.ndarray] = []
        if mode in ("spatial", "both"):
            for _ in range(self.num_spatial_targets):
                targets.append(self._sector_tube())
        if mode in ("temporal", "both"):
            targets.append(self._future_frames())

        union = np.unique(np.concatenate(targets))
        all_idx = np.arange(self.T * self.per_frame)
        ctx = np.setdiff1d(all_idx, union)
        if ctx.size == 0:  # degenerate — fall back to "everything but the union"
            ctx = np.setdiff1d(all_idx, union[: union.size // 2])
        return ctx, [np.ascontiguousarray(t) for t in targets]

    # ---- for visualization -------------------------------------------------- #
    def role_grid(self, ctx_idx, target_list) -> np.ndarray:
        """(T, GH, GW) int grid: 0=context, 1..=target-block id (for plotting)."""
        g = np.full(self.T * self.per_frame, -1, dtype=int)
        g[ctx_idx] = 0
        for b, t in enumerate(target_list, start=1):
            g[t] = b
        return g.reshape(self.T, self.GH, self.GW)


def build_mask_sampler(cfg_mask: dict, grid_thw, seed: int | None = None) -> MaskSampler:
    """Instantiate from the ``mask:`` config section."""
    return MaskSampler(
        grid_thw=grid_thw,
        mode=cfg_mask.get("mode", "mixed"),
        sector_width=cfg_mask.get("sector_width", (3, 6)),
        num_spatial_targets=cfg_mask.get("num_spatial_targets", 2),
        n_future=cfg_mask.get("n_future", 1),
        seed=seed,
    )
