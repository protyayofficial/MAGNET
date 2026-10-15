"""Core neural modules and manifold adapters for GDT."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _infer_nodes_from_strict_lower_dim(n_features: int) -> int:
    """Infer matrix size ``d`` from strict-lower vector size ``m = d(d-1)/2``."""
    disc = 1 + 8 * n_features
    root = int(math.sqrt(disc))
    if root * root != disc:
        return -1
    d = (1 + root) // 2
    if d * (d - 1) // 2 != n_features:
        return -1
    return d


class CorrCholeskyAdapter:
    """Map between latent strict-lower vectors and correlation matrices.

    The latent vector stores the strict lower-triangular coefficients of a
    unit-diagonal Cholesky-like factor. ``z_mean`` and ``z_scale`` are used to
    undo standardization before reconstructing correlation matrices.
    """

    def __init__(
        self,
        n_features: int,
        z_mean: Optional[np.ndarray] = None,
        z_scale: Optional[np.ndarray] = None,
        eps: float = 1e-8,
    ):
        self.n_features = n_features
        self.n_nodes = _infer_nodes_from_strict_lower_dim(n_features)
        if self.n_nodes < 2:
            raise ValueError("Invalid strict-lower feature dimension.")
        self.eps = eps

        self.z_mean = (
            np.zeros(n_features, dtype=np.float32)
            if z_mean is None
            else np.asarray(z_mean, dtype=np.float32)
        )
        self.z_scale = (
            np.ones(n_features, dtype=np.float32)
            if z_scale is None
            else np.clip(np.asarray(z_scale, dtype=np.float32), eps, None)
        )

        tril_i, tril_j = np.tril_indices(self.n_nodes, k=-1)
        self.tril_i = tril_i.astype(np.int64)
        self.tril_j = tril_j.astype(np.int64)
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}

    def _tensors(self, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        key = f"{device}_{dtype}"
        if key not in self._cache:
            self._cache[key] = {
                "eye": torch.eye(self.n_nodes, device=device, dtype=dtype),
                "mean": torch.from_numpy(self.z_mean).to(device=device, dtype=dtype),
                "scale": torch.from_numpy(self.z_scale).to(device=device, dtype=dtype),
                "tril_i": torch.from_numpy(self.tril_i).to(device=device),
                "tril_j": torch.from_numpy(self.tril_j).to(device=device),
            }
        return self._cache[key]

    def x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        """Convert standardized latent ``x`` to unstandardized latent ``z``."""
        c = self._tensors(x.device, x.dtype)
        return x * c["scale"] + c["mean"]

    def z_to_x(self, z: torch.Tensor) -> torch.Tensor:
        """Convert unstandardized latent ``z`` to standardized latent ``x``."""
        c = self._tensors(z.device, z.dtype)
        return (z - c["mean"]) / c["scale"]

    def z_to_corr(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct correlation matrices from unstandardized latent vectors."""
        c = self._tensors(z.device, z.dtype)
        original_shape = z.shape[:-1]
        z = z.reshape(-1, z.shape[-1])
        batch_size = z.shape[0]

        L = c["eye"].unsqueeze(0).expand(batch_size, -1, -1).clone()
        L[:, c["tril_i"], c["tril_j"]] = z
        gram = L @ L.transpose(-1, -2)

        diag = gram.diagonal(dim1=-2, dim2=-1).clamp(min=self.eps)
        inv_sqrt = torch.rsqrt(diag)
        corr = inv_sqrt.unsqueeze(-1) * gram * inv_sqrt.unsqueeze(-2)
        corr = 0.5 * (corr + corr.transpose(-1, -2))
        idx = torch.arange(self.n_nodes, device=z.device)
        corr[:, idx, idx] = 1.0
        return corr.reshape(*original_shape, self.n_nodes, self.n_nodes)

    def x_to_corr(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct correlation matrices from standardized latent vectors."""
        return self.z_to_corr(self.x_to_z(x))

    def x_to_corr_np(self, x: np.ndarray) -> np.ndarray:
        """NumPy wrapper for ``x_to_corr``."""
        with torch.no_grad():
            x = np.asarray(x, dtype=np.float32)
            tensor = torch.from_numpy(x)
            return self.x_to_corr(tensor).numpy().astype(np.float64)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for scalar diffusion times."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype)
            * (-math.log(10000.0) / max(half, 1))
        )
        args = t[:, None] * freq[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AnatomicalPositionalEncoding(nn.Module):
    """Learned region-wise positional encoding."""

    def __init__(self, n_nodes: int, hidden_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, n_nodes, hidden_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed


class GeGLU(nn.Module):
    """Gated GELU activation used in transformer feed-forward blocks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj, gate = x.chunk(2, dim=-1)
        return x_proj * F.gelu(gate)


class QKNormMultiheadAttention(nn.Module):
    """Multi-head attention with layer-normalized query/key vectors."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, _ = x.shape

        q = self.q_proj(x).view(batch_size, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        out = (attention @ v).transpose(1, 2).reshape(batch_size, n_tokens, self.embed_dim)
        return self.out_proj(out)


class TransformerEncoderLayerWithQKNorm(nn.Module):
    """Transformer encoder layer with optional pre-norm and QK-normalized attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_pre_norm: bool = True,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.use_pre_norm = use_pre_norm

        if use_qk_norm:
            self.self_attn = QKNormMultiheadAttention(d_model, nhead, dropout)
            self._uses_builtin_mha = False
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )
            self._uses_builtin_mha = True

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward * 2),
            GeGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        if self._uses_builtin_mha:
            out, _ = self.self_attn(x, x, x, need_weights=False)
            return out
        return self.self_attn(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_pre_norm:
            x = x + self.dropout1(self._attention(self.norm1(x)))
            x = x + self.dropout2(self.feed_forward(self.norm2(x)))
            return x

        x = self.norm1(x + self.dropout1(self._attention(x)))
        x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x


def batch_diversity_loss(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Diversity regularizer based on average pairwise sample distance."""
    if x.dim() != 2 or x.size(0) < 2:
        return x.new_tensor(0.0)

    x_centered = x - x.mean(dim=0, keepdim=True)
    x_norm = x_centered / (x_centered.norm(dim=1, keepdim=True) + eps)
    gram = x_norm @ x_norm.T
    pairwise_sq = (2.0 - 2.0 * gram).clamp(min=0.0)

    mask = ~torch.eye(x.size(0), device=x.device, dtype=torch.bool)
    return pairwise_sq[mask].mean()


__all__ = [
    "CorrCholeskyAdapter",
    "SinusoidalTimeEmbedding",
    "AnatomicalPositionalEncoding",
    "QKNormMultiheadAttention",
    "TransformerEncoderLayerWithQKNorm",
    "GeGLU",
    "batch_diversity_loss",
]
