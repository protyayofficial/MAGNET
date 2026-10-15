"""Conditioning modules used during GDT denoising and sampling."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn
import torch.nn.functional as F


class DiversityPreservingCFG:
    """Classifier-free guidance that preserves non-discriminative variance.

    Standard classifier-free guidance scales the conditional direction
    ``(pred_cond - pred_uncond)`` uniformly, which can reduce sample diversity.
    This variant keeps an orthogonal residual from ``pred_uncond`` to preserve
    variability not aligned with the class-discriminative direction.
    """

    def __init__(self, base_scale: float = 1.5, diversity_weight: float = 0.3):
        self.base_scale = float(base_scale)
        self.diversity_weight = float(diversity_weight)

    def apply(
        self,
        pred_cond: torch.Tensor,
        pred_uncond: torch.Tensor,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply diversity-preserving classifier-free guidance."""
        scale = float(self.base_scale if scale is None else scale)

        direction = pred_cond - pred_uncond
        direction_norm = torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-8)
        direction_unit = direction / direction_norm

        projection = (pred_uncond * direction_unit).sum(dim=-1, keepdim=True) * direction_unit
        orthogonal = pred_uncond - projection

        # Confidence-weighted guidance reduces over-steering when the conditional
        # direction has low magnitude.
        confidence = torch.tanh(direction_norm)
        guided = pred_uncond + (scale * confidence) * direction
        return guided + self.diversity_weight * orthogonal


class SelfConditionedSpectralModule(nn.Module):
    """Spectral feature encoder for self-conditioned diffusion.

    Input ``x`` is the flattened strict-lower representation of a matrix.
    The module supports two feature modes:
    - ``proxy_v0``: low-cost distributional moments on latent entries.
    - ``proxy_v1``: intrinsic log-spectrum/connectivity invariants computed
      from the decoded correlation matrix.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_nodes: int,
        n_spectral_features: int = 16,
        feature_mode: str = "proxy_v0",
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_features = n_nodes * (n_nodes - 1) // 2
        self.n_spectral = n_spectral_features
        self.feature_mode = str(feature_mode)
        if self.feature_mode not in {"proxy_v0", "proxy_v1"}:
            raise ValueError(f"Unknown spectral feature mode: {self.feature_mode}")

        tril = torch.tril_indices(row=n_nodes, col=n_nodes, offset=-1)
        self.register_buffer("tril_i", tril[0].to(torch.long))
        self.register_buffer("tril_j", tril[1].to(torch.long))

        self.spectral_embed = nn.Sequential(
            nn.Linear(n_spectral_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.null_embedding = nn.Parameter(torch.zeros(hidden_dim))

    def _compute_proxy_features(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy v0 proxy features from latent entry statistics."""
        batch_size = x.shape[0]
        features = torch.zeros(batch_size, self.n_spectral, device=x.device, dtype=x.dtype)

        mean = x.mean(dim=-1)
        std = x.std(dim=-1).clamp(min=1e-8)
        centered = x - mean.unsqueeze(-1)

        features[:, 0] = mean
        features[:, 1] = std
        features[:, 2] = torch.abs(x).sum(dim=-1)
        features[:, 3] = torch.abs(x).mean(dim=-1)
        features[:, 4] = x.max(dim=-1).values
        features[:, 5] = x.min(dim=-1).values
        features[:, 6] = torch.abs(x).max(dim=-1).values
        features[:, 7] = (x > 0).float().mean(dim=-1)
        features[:, 8] = (torch.abs(x) < 0.1).float().mean(dim=-1)
        features[:, 9] = (x**2).mean(dim=-1)
        features[:, 10] = (centered**3).mean(dim=-1) / (std**3 + 1e-8)
        features[:, 11] = (centered**4).mean(dim=-1) / (std**4 + 1e-8) - 3.0
        features[:, 12] = (x**3).mean(dim=-1)
        features[:, 13] = (x**4).mean(dim=-1)
        features[:, 14] = (x > mean.unsqueeze(-1)).float().sum(dim=-1)
        features[:, 15] = x.max(dim=-1).values - x.min(dim=-1).values

        # Soft clipping keeps large moments numerically stable.
        return torch.tanh(features / 10.0) * 10.0

    def _decode_corr_from_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Decode strict-lower latent vectors into correlation matrices."""
        x = torch.nan_to_num(x, nan=0.0, posinf=6.0, neginf=-6.0)
        bsz = x.shape[0]
        L = torch.eye(self.n_nodes, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(bsz, 1, 1)
        L[:, self.tril_i, self.tril_j] = x

        gram = L @ L.transpose(-1, -2)
        diag = torch.diagonal(gram, dim1=-2, dim2=-1).clamp_min(1e-6)
        inv_sqrt_diag = torch.rsqrt(diag)
        corr = gram * inv_sqrt_diag.unsqueeze(-1) * inv_sqrt_diag.unsqueeze(-2)
        corr = 0.5 * (corr + corr.transpose(-1, -2))
        return torch.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)

    def _compute_proxy_v1_features(self, x: torch.Tensor) -> torch.Tensor:
        """v1 features from intrinsic correlation-spectrum invariants."""
        corr = self._decode_corr_from_latent(x)
        eigvals = torch.linalg.eigvalsh(corr).clamp_min(1e-8)
        log_eig = torch.log(eigvals)
        bsz = x.shape[0]
        n = float(self.n_nodes)

        features = torch.zeros(bsz, self.n_spectral, device=x.device, dtype=x.dtype)
        log_mean = log_eig.mean(dim=-1)
        log_std = log_eig.std(dim=-1).clamp(min=1e-8)
        centered = log_eig - log_mean.unsqueeze(-1)

        features[:, 0] = log_mean
        features[:, 1] = log_std
        features[:, 2] = (centered**3).mean(dim=-1) / (log_std**3 + 1e-8)
        features[:, 3] = (centered**4).mean(dim=-1) / (log_std**4 + 1e-8) - 3.0
        features[:, 4] = log_eig.min(dim=-1).values
        features[:, 5] = log_eig.max(dim=-1).values

        p = eigvals / eigvals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=-1)
        features[:, 6] = entropy
        features[:, 7] = torch.exp(entropy) / n
        eig_sq_sum = (eigvals**2).sum(dim=-1).clamp_min(1e-8)
        features[:, 8] = (eigvals.sum(dim=-1) ** 2) / (n * eig_sq_sum)
        eig_max = eigvals.max(dim=-1).values.clamp_min(1e-8)
        features[:, 9] = eig_sq_sum / (eig_max**2 + 1e-8)
        features[:, 10] = torch.log(eig_max) - torch.log(eigvals.min(dim=-1).values.clamp_min(1e-8))
        features[:, 11] = eigvals[:, -1] - eigvals[:, -2]

        eye = torch.eye(self.n_nodes, device=x.device, dtype=x.dtype).unsqueeze(0)
        off_mask = 1.0 - eye
        off = (corr - eye) * off_mask
        denom = n * (n - 1.0)
        features[:, 12] = torch.abs(off).sum(dim=(-1, -2)) / denom
        features[:, 13] = torch.sqrt((off**2).sum(dim=(-1, -2)) / denom + 1e-8)
        features[:, 14] = ((off > 0).to(x.dtype) * off_mask).sum(dim=(-1, -2)) / denom
        features[:, 15] = ((off < 0).to(x.dtype) * off_mask).sum(dim=(-1, -2)) / denom

        return torch.tanh(features / 10.0) * 10.0

    def compute_spectral_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spectral features from flattened connectivity vectors."""
        if self.feature_mode == "proxy_v1":
            return self._compute_proxy_v1_features(x)
        return self._compute_proxy_features(x)

    def compute_features_for_mode(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Compute raw proxy features for a requested mode."""
        if mode == "proxy_v1":
            return self._compute_proxy_v1_features(x)
        if mode == "proxy_v0":
            return self._compute_proxy_features(x)
        raise ValueError(f"Unknown spectral feature mode: {mode}")

    def embed_features(self, features: torch.Tensor) -> torch.Tensor:
        """Map raw spectral features to the model hidden dimension."""
        return self.spectral_embed(features)

    def null(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return the learnable null spectral embedding."""
        return self.null_embedding.unsqueeze(0).expand(batch_size, -1)

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode spectral features or return a learnable null embedding."""
        if x is None:
            if batch_size is None or device is None:
                raise ValueError("batch_size and device are required when x is None")
            return self.null(batch_size, device)

        return self.embed_features(self.compute_spectral_features(x))


def compute_spectral_consistency_loss(
    x0_pred: torch.Tensor,
    x0_true: torch.Tensor,
    spectral_module: SelfConditionedSpectralModule,
) -> torch.Tensor:
    """Backward-compatible mean-reduced spectral consistency loss."""
    return compute_spectral_consistency_loss_with_reduction(
        x0_pred,
        x0_true,
        spectral_module,
        reduction="mean",
    )


def compute_spectral_consistency_loss_with_reduction(
    x0_pred: torch.Tensor,
    x0_true: torch.Tensor,
    spectral_module: SelfConditionedSpectralModule,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """Per-sample or mean spectral consistency loss."""
    pred_features = spectral_module.compute_spectral_features(x0_pred)
    true_features = spectral_module.compute_spectral_features(x0_true)
    per_sample = torch.mean((pred_features - true_features) ** 2, dim=-1)
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"Unknown reduction: {reduction}")


__all__ = [
    "DiversityPreservingCFG",
    "SelfConditionedSpectralModule",
    "compute_spectral_consistency_loss",
    "compute_spectral_consistency_loss_with_reduction",
]
