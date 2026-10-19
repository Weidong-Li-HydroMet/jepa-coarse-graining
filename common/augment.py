"""Climate-aware multi-crop augmentations (DINO only — MAE/I-JEPA use masking, no augs).

No color jitter. The view menu (config-listed in dino/config.yaml):
  * cyclic longitude roll  (the tropical band is periodic in lon — per-sample shift)
  * lat/lon crop + resize back to the full grid (global = large area, local = small)
  * additive Gaussian noise (inputs are standardized anomalies, so std is O(1))
  * per-channel intensity scaling
  * channel dropout (zero one of OLR/U850/U200; never on global views)

All ops act on a batched [B, C, H, W] tensor. Simplification (v1): the crop box is
shared across the batch within one view (it still differs between views and steps);
the cyclic lon roll is per-sample.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class MultiCropClimate:
    """Produce DINO global/local views from a batch of maps [B, C, H, W]."""

    def __init__(
        self,
        n_global: int = 2,
        n_local: int = 6,
        global_area: tuple[float, float] = (0.5, 1.0),
        local_area: tuple[float, float] = (0.15, 0.5),
        noise_std: float = 0.05,
        chan_scale: tuple[float, float] = (0.9, 1.1),
        chan_drop_p: float = 0.1,
        generator: torch.Generator | None = None,
    ):
        self.n_global = n_global
        self.n_local = n_local
        self.global_area = global_area
        self.local_area = local_area
        self.noise_std = noise_std
        self.chan_scale = chan_scale
        self.chan_drop_p = chan_drop_p
        self.gen = generator

    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen)

    def _cyclic_roll(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample random cyclic shift along longitude (last dim)."""
        B, C, H, W = x.shape
        shifts = torch.randint(0, W, (B,), generator=self.gen, device=x.device)
        idx = (torch.arange(W, device=x.device)[None, :] + shifts[:, None]) % W
        return x.gather(-1, idx[:, None, None, :].expand(B, C, H, W))

    def _crop_resize(self, x: torch.Tensor, area_range: tuple[float, float]) -> torch.Tensor:
        """Crop a lat/lon box covering `area` fraction of the map, resize back."""
        B, C, H, W = x.shape
        area = area_range[0] + (area_range[1] - area_range[0]) * self._rand(1).item()
        # split the area between the two axes; keep lat fraction >= 0.5 so a local
        # view never collapses below 4 rows (H=32 -> crop height >= 16 > patch 4)
        frac = area ** 0.5
        fh = max(frac, 0.5)
        fw = max(area / fh, 4.0 / W)
        ch, cw = max(int(round(fh * H)), 4), max(int(round(fw * W)), 4)
        top = int(self._rand(1).item() * (H - ch + 1))
        left = int(self._rand(1).item() * (W - cw + 1))
        crop = x[:, :, top : top + ch, left : left + cw]
        return F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)

    def _photometric(self, x: torch.Tensor, allow_chan_drop: bool) -> torch.Tensor:
        B, C, _, _ = x.shape
        lo, hi = self.chan_scale
        scale = lo + (hi - lo) * self._rand(B, C, 1, 1).to(x.device)
        x = x * scale
        if self.noise_std > 0:
            x = x + self.noise_std * torch.randn(
                x.shape, generator=self.gen, device=x.device
            )
        if allow_chan_drop and self.chan_drop_p > 0:
            keep = (self._rand(B, C, 1, 1).to(x.device) > self.chan_drop_p).float()
            # never drop all channels of a sample
            keep = torch.where(
                keep.sum(dim=1, keepdim=True) == 0, torch.ones_like(keep), keep
            )
            x = x * keep
        return x

    def _one_view(self, x: torch.Tensor, area_range, allow_chan_drop: bool) -> torch.Tensor:
        v = self._cyclic_roll(x)
        v = self._crop_resize(v, area_range)
        return self._photometric(v, allow_chan_drop)

    def __call__(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """x: [B, C, H, W] -> (global views, local views), each a list of [B, C, H, W]."""
        globals_ = [self._one_view(x, self.global_area, False) for _ in range(self.n_global)]
        locals_ = [self._one_view(x, self.local_area, True) for _ in range(self.n_local)]
        return globals_, locals_


def build_augment(cfg: dict, generator: torch.Generator | None = None) -> MultiCropClimate:
    """Instantiate from the `augment:` section of dino/config.yaml."""
    return MultiCropClimate(
        n_global=cfg["n_global"],
        n_local=cfg["n_local"],
        global_area=tuple(cfg["global_area"]),
        local_area=tuple(cfg["local_area"]),
        noise_std=cfg["noise_std"],
        chan_scale=tuple(cfg["chan_scale"]),
        chan_drop_p=cfg["chan_drop_p"],
        generator=generator,
    )
