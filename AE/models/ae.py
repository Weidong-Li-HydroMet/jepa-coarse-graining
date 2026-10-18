"""PhysAnchor-MJO-AE: a plain reconstruction autoencoder, matched to V-JEPA.

From a 3-day clip ``(B, C, T, H, W)`` the shared V-JEPA ``STViTBackbone`` encoder produces
patch tokens; we pool them to a single per-clip latent ``(B, D)`` (D = the matched V-JEPA
``embed_dim``, NOT the paper's 1024), and a lightweight decoder reconstructs the target
day's field ``(B, C, H, W)``.  Loss = MSE (or smooth-L1) over the reconstructed field.

This is intentionally a PLAIN AE — **no masking, no EMA target, no predictor** — so it is
the reconstruction-baseline counterpart to ``jepa.models.vjepa.VJEPA``.  The encoder is the
exact same class/architecture V-JEPA uses, so the two latent spaces are directly
comparable (same encoder, same latent dim, same input samples).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa.models.encoder import STViTBackbone
from AE.models.decoder import SpatialDecoder


class MJOAutoencoder(nn.Module):
    def __init__(
        self,
        encoder: STViTBackbone,
        decoder: SpatialDecoder,
        pool: str = "current",       # 'current' (last-day tokens, == V-JEPA embed_clip) | 'all'
        target_day: str = "last",    # 'last' (day t, the paper) | 'center' (day t-1)
        loss_type: str = "mse",      # 'mse' (paper) | 'smooth_l1'
    ):
        super().__init__()
        # Hard constraint: latent dim == encoder embed_dim (the matched V-JEPA dim).
        assert decoder.proj_in.in_features == encoder.embed_dim, (
            f"latent_dim ({decoder.proj_in.in_features}) must equal encoder.embed_dim "
            f"({encoder.embed_dim}) for an apples-to-apples comparison with V-JEPA"
        )
        self.encoder = encoder
        self.decoder = decoder
        self.pool = pool
        self.target_day = target_day
        self.loss_type = loss_type

    # ---- latent ------------------------------------------------------------- #
    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, H, W)`` -> per-clip latent ``(B, D)`` (same pooling as V-JEPA)."""
        tokens = self.encoder(x)  # (B, L, D)
        if self.pool == "all":
            return tokens.mean(dim=1)
        idx = torch.from_numpy(
            self.encoder.frame_token_index(self.encoder.num_frames - 1)
        ).long().to(x.device)
        return tokens[:, idx].mean(dim=1)  # last-day (day-t) token pool

    def _target_index(self) -> int:
        T = self.encoder.num_frames
        return T - 1 if self.target_day == "last" else T // 2

    def _recon_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(recon, target)
        return F.mse_loss(recon, target)

    # ---- forward ------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """x: ``(B, C, T, H, W)`` -> (scalar reconstruction loss, diagnostics dict)."""
        z = self.encode_latent(x)
        recon = self.decoder(z)                  # (B, C, H, W)
        target = x[:, :, self._target_index()]   # (B, C, H, W)
        loss = self._recon_loss(recon, target)
        with torch.no_grad():
            # variance of the target-field anomalies = the trivial-MSE baseline (predict 0)
            base = target.float().pow(2).mean()
            stats = {
                "loss": float(loss.detach()),
                "recon_loss": float(loss.detach()),
                "rel_mse": float((loss.detach() / (base + 1e-8))),  # <1 means beats predict-zero
                "lat_std": float(z.reshape(-1, z.shape[-1]).std(0).mean()),  # latent spread monitor
            }
        return loss, stats

    # ---- downstream embedding (mirrors V-JEPA.embed_clip signature) --------- #
    @property
    def probe_encoder(self) -> STViTBackbone:
        return self.encoder

    @torch.no_grad()
    def embed_clip(self, x: torch.Tensor, pool: str | None = None) -> torch.Tensor:
        """Per-clip latent ``(B, D)``; defaults to the training pooling."""
        enc = self.encoder
        tokens = enc(x)
        p = pool or self.pool
        if p == "all":
            return tokens.mean(dim=1)
        idx = torch.from_numpy(enc.frame_token_index(enc.num_frames - 1)).long().to(x.device)
        return tokens[:, idx].mean(dim=1)


def build_ae(encoder, decoder, cfg_method: dict) -> MJOAutoencoder:
    """Instantiate from the ``method:`` config section."""
    return MJOAutoencoder(
        encoder,
        decoder,
        pool=cfg_method.get("pool", "current"),
        target_day=cfg_method.get("target_day", "last"),
        loss_type=cfg_method.get("loss_type", "mse"),
    )
