"""Training and sampling pipeline for the manifold-aware GDT prior."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

from .model import (
    CorrCholeskyAdapter,
    SinusoidalTimeEmbedding,
    AnatomicalPositionalEncoding,
    TransformerEncoderLayerWithQKNorm,
)

from .conditioning import (
    DiversityPreservingCFG,
    SelfConditionedSpectralModule,
    compute_spectral_consistency_loss,
)


def _anchor_moment_loss(
    x0_pred: torch.Tensor,
    x0_true: torch.Tensor,
    anchor_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Match first two moments on anchor dimensions while lightly preserving non-anchor variance."""
    anchor_mask = anchor_mask.clamp(0.0, 1.0)
    non_anchor_mask = 1.0 - anchor_mask

    def _weighted_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (w * x).sum(dim=1, keepdim=True) / (w.sum(dim=1, keepdim=True) + eps)

    def _weighted_std(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        mean = _weighted_mean(x, w)
        var = (w * (x - mean) ** 2).sum(dim=1, keepdim=True) / (w.sum(dim=1, keepdim=True) + eps)
        return torch.sqrt(var + eps)

    mu_pred_anchor = _weighted_mean(x0_pred, anchor_mask)
    mu_true_anchor = _weighted_mean(x0_true, anchor_mask)
    std_pred_anchor = _weighted_std(x0_pred, anchor_mask)
    std_true_anchor = _weighted_std(x0_true, anchor_mask)

    std_pred_non_anchor = _weighted_std(x0_pred, non_anchor_mask)
    std_true_non_anchor = _weighted_std(x0_true, non_anchor_mask)

    anchor_term = (mu_pred_anchor - mu_true_anchor).abs().mean() + (std_pred_anchor - std_true_anchor).abs().mean()
    non_anchor_term = (std_pred_non_anchor - std_true_non_anchor).abs().mean()
    return anchor_term + 0.2 * non_anchor_term


def _spd_margin_loss(corr: torch.Tensor, margin: float = 1e-3) -> torch.Tensor:
    """Penalize decoded correlations that approach the SPD boundary."""
    evals = torch.linalg.eigvalsh(0.5 * (corr + corr.transpose(-1, -2)))
    return torch.relu(margin - evals[:, 0]).mean()


# =============================================================================
# Transformer Denoiser
# =============================================================================

class GraphDiffusionTransformer(nn.Module):
    """Graph transformer denoiser used in the diffusion prior."""

    def __init__(
        self,
        n_nodes: int,
        n_conditions: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_pre_norm: bool = True,
        use_qk_norm: bool = True,
        use_spectral_cond: bool = True,
        spectral_feature_mode: str = "amortized_bridge_v1",
    ):
        super().__init__()
        valid_modes = {"amortized_bridge_v1", "self_cond_v1", "teacher_bridge_v1"}
        if spectral_feature_mode not in valid_modes:
            raise ValueError(
                "spectral_feature_mode must be one of "
                f"{sorted(valid_modes)}, got {spectral_feature_mode!r}."
            )
        self.n_nodes = n_nodes
        self.n_features = n_nodes * (n_nodes - 1) // 2
        self.hidden_dim = hidden_dim
        self.use_spectral_cond = use_spectral_cond
        self.spectral_feature_mode = spectral_feature_mode
        self.amortized_bridge_mode = spectral_feature_mode == "amortized_bridge_v1"
        self.teacher_bridge_mode = spectral_feature_mode == "teacher_bridge_v1"
        self.self_condition_mode = spectral_feature_mode == "self_cond_v1"

        # Indices for lower triangular
        tril_i, tril_j = np.tril_indices(n_nodes, k=-1)
        self.register_buffer("tril_i", torch.from_numpy(tril_i.astype(np.int64)))
        self.register_buffer("tril_j", torch.from_numpy(tril_j.astype(np.int64)))

        # Input projection
        self.input_proj = nn.Linear(n_nodes, hidden_dim)

        # Anatomical positional encoding
        self.pos_encoding = AnatomicalPositionalEncoding(n_nodes, hidden_dim)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Condition embedding (GMM subtypes + uncond)
        self.cond_embed = nn.Embedding(n_conditions, hidden_dim)

        # Optional self-conditioning on spectral statistics.
        if use_spectral_cond:
            feature_mode = "proxy_v1"
            self.spectral_module = SelfConditionedSpectralModule(hidden_dim, n_nodes, feature_mode=feature_mode)
            self.spectral_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            if self.amortized_bridge_mode:
                bridge_input_dim = self.spectral_module.n_spectral + 2 * hidden_dim
                self.bridge_predictor = nn.Sequential(
                    nn.Linear(bridge_input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.spectral_module.n_spectral),
                )

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerWithQKNorm(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                use_pre_norm=use_pre_norm,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim) if use_pre_norm else None

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, n_nodes)

        # Better initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with better defaults for diffusion."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _embed_condition(self, cond: torch.Tensor) -> torch.Tensor:
        """Embed hard condition tokens."""
        return self.cond_embed(cond)

    def compute_context_embeddings(
        self,
        cond: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return time and condition embeddings used by denoiser and auxiliaries."""
        t_emb = self.time_mlp(self.time_embed(t))
        c_emb = self._embed_condition(cond)
        return t_emb, c_emb

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        spectral_cond: Optional[torch.Tensor] = None,  # Pre-computed spectral embedding
    ) -> torch.Tensor:
        """
        Args:
            x: [B, n_features] noisy latent vectors
            cond: [B] condition indices
            t: [B] normalized timestep
            spectral_cond: [B, hidden_dim] pre-computed spectral embedding (from self-conditioning)
        Returns:
            pred: [B, n_features] predicted velocity/noise
        """
        B = x.shape[0]

        # Build symmetric adjacency from lower triangular
        adj = torch.zeros(B, self.n_nodes, self.n_nodes, device=x.device, dtype=x.dtype)
        adj[:, self.tril_i, self.tril_j] = x
        adj[:, self.tril_j, self.tril_i] = x

        # Node features
        h = self.input_proj(adj)
        h = self.pos_encoding(h)

        # Time and condition embedding
        t_emb, c_emb = self.compute_context_embeddings(cond, t)
        combined_cond = t_emb + c_emb

        # Self-Conditioned Spectral Features
        if self.use_spectral_cond:
            if spectral_cond is None:
                # Use null embedding (no spectral info)
                spectral_cond = self.spectral_module(None, batch_size=B, device=x.device)

            # Gated addition
            gate = self.spectral_gate(spectral_cond)
            combined_cond = combined_cond + gate * spectral_cond

        h = h + combined_cond.unsqueeze(1)

        # Transformer
        for layer in self.transformer_layers:
            h = layer(h)
        if self.final_norm is not None:
            h = self.final_norm(h)

        # Output
        out = self.out_proj(h)
        out = 0.5 * (out + out.transpose(-1, -2))
        return out[:, self.tril_i, self.tril_j]

    def predict_amortized_bridge_features(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict raw proxy_v1 bridge features from cheap proxy_v0 summaries."""
        if not self.amortized_bridge_mode:
            raise RuntimeError("Amortized bridge predictor is only available in amortized_bridge_v1 mode.")
        t_emb, c_emb = self.compute_context_embeddings(cond, t)
        proxy_v0 = self.spectral_module.compute_features_for_mode(x, "proxy_v0")
        bridge_in = torch.cat([proxy_v0, t_emb, c_emb], dim=-1)
        raw = self.bridge_predictor(bridge_in)
        return torch.tanh(raw / 10.0) * 10.0

    def get_x0_from_v_pred(
        self,
        x_t: torch.Tensor,
        v_pred: torch.Tensor,
        alpha_bar: torch.Tensor,
    ) -> torch.Tensor:
        """Convert v-prediction to x0 estimate."""
        return torch.sqrt(alpha_bar) * x_t - torch.sqrt(1 - alpha_bar) * v_pred


class VectorDiffusionMLP(nn.Module):
    """Flat vector-space denoiser used as a non-graph backbone ablation."""

    def __init__(
        self,
        n_nodes: int,
        n_conditions: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,  # kept for interface compatibility
        dropout: float = 0.1,
        use_pre_norm: bool = True,
        use_qk_norm: bool = True,  # kept for interface compatibility
        use_spectral_cond: bool = True,
        spectral_feature_mode: str = "amortized_bridge_v1",
    ):
        super().__init__()
        del num_heads, use_qk_norm
        valid_modes = {"amortized_bridge_v1", "self_cond_v1", "teacher_bridge_v1"}
        if spectral_feature_mode not in valid_modes:
            raise ValueError(
                "spectral_feature_mode must be one of "
                f"{sorted(valid_modes)}, got {spectral_feature_mode!r}."
            )
        self.n_nodes = n_nodes
        self.n_features = n_nodes * (n_nodes - 1) // 2
        self.hidden_dim = hidden_dim
        self.use_spectral_cond = use_spectral_cond
        self.spectral_feature_mode = spectral_feature_mode
        self.amortized_bridge_mode = spectral_feature_mode == "amortized_bridge_v1"
        self.teacher_bridge_mode = spectral_feature_mode == "teacher_bridge_v1"
        self.self_condition_mode = spectral_feature_mode == "self_cond_v1"
        self.use_pre_norm = use_pre_norm

        self.input_proj = nn.Linear(self.n_features, hidden_dim)

        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_embed = nn.Embedding(n_conditions, hidden_dim)

        if use_spectral_cond:
            feature_mode = "proxy_v1"
            self.spectral_module = SelfConditionedSpectralModule(hidden_dim, n_nodes, feature_mode=feature_mode)
            self.spectral_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            if self.amortized_bridge_mode:
                bridge_input_dim = self.spectral_module.n_spectral + 2 * hidden_dim
                self.bridge_predictor = nn.Sequential(
                    nn.Linear(bridge_input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.spectral_module.n_spectral),
                )

        self.blocks = nn.ModuleList([
            nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(hidden_dim),
                    "ff": nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim * 4, hidden_dim),
                        nn.Dropout(dropout),
                    ),
                }
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim) if use_pre_norm else None
        self.out_proj = nn.Linear(hidden_dim, self.n_features)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _embed_condition(self, cond: torch.Tensor) -> torch.Tensor:
        return self.cond_embed(cond)

    def compute_context_embeddings(
        self,
        cond: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_emb = self.time_mlp(self.time_embed(t))
        c_emb = self._embed_condition(cond)
        return t_emb, c_emb

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        spectral_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = x.shape[0]

        h = self.input_proj(x)
        t_emb, c_emb = self.compute_context_embeddings(cond, t)
        combined_cond = t_emb + c_emb

        if self.use_spectral_cond:
            if spectral_cond is None:
                spectral_cond = self.spectral_module(None, batch_size=bsz, device=x.device)
            gate = self.spectral_gate(spectral_cond)
            combined_cond = combined_cond + gate * spectral_cond

        h = h + combined_cond
        for block in self.blocks:
            if self.use_pre_norm:
                h = h + block["ff"](block["norm"](h))
            else:
                h = block["norm"](h + block["ff"](h))
        if self.final_norm is not None:
            h = self.final_norm(h)
        return self.out_proj(h)

    def predict_amortized_bridge_features(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if not self.amortized_bridge_mode:
            raise RuntimeError("Amortized bridge predictor is only available in amortized_bridge_v1 mode.")
        t_emb, c_emb = self.compute_context_embeddings(cond, t)
        proxy_v0 = self.spectral_module.compute_features_for_mode(x, "proxy_v0")
        bridge_in = torch.cat([proxy_v0, t_emb, c_emb], dim=-1)
        raw = self.bridge_predictor(bridge_in)
        return torch.tanh(raw / 10.0) * 10.0

    def get_x0_from_v_pred(
        self,
        x_t: torch.Tensor,
        v_pred: torch.Tensor,
        alpha_bar: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sqrt(alpha_bar) * x_t - torch.sqrt(1 - alpha_bar) * v_pred


def _build_denoiser(
    cfg: "GDTConfig",
    n_nodes: int,
    n_conditions: int,
) -> nn.Module:
    common_kwargs = dict(
        n_nodes=n_nodes,
        n_conditions=n_conditions,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        use_pre_norm=cfg.use_pre_norm,
        use_qk_norm=cfg.use_qk_norm,
        use_spectral_cond=cfg.use_spectral_cond,
        spectral_feature_mode=cfg.spectral_feature_mode,
    )
    if cfg.denoiser_arch == "graph_transformer":
        return GraphDiffusionTransformer(**common_kwargs)
    if cfg.denoiser_arch == "vector_mlp":
        return VectorDiffusionMLP(**common_kwargs)
    raise ValueError(
        f"Unsupported denoiser_arch={cfg.denoiser_arch!r}. "
        "Expected one of ['graph_transformer', 'vector_mlp']."
    )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GDTConfig:
    """Hyperparameters for GDT prior training and sampling."""

    # Architecture
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    use_pre_norm: bool = True
    use_qk_norm: bool = True
    denoiser_arch: str = "graph_transformer"

    # Spectral conditioning
    use_spectral_cond: bool = True
    self_cond_prob: float = 0.5
    spectral_loss_weight: float = 0.1
    spectral_feature_mode: str = "amortized_bridge_v1"  # {"amortized_bridge_v1", "self_cond_v1", "teacher_bridge_v1"}
    amortized_bridge_loss_weight: float = 0.1
    anchor_loss_weight: float = 0.0

    # Training
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_epochs: int = 5

    # Diffusion
    diffusion_steps: int = 1000
    ddim_steps: int = 6
    ddim_eta: float = 0.05
    schedule: str = "cosine"
    v_prediction: bool = True
    clip_sample: float = 6.0
    use_min_snr: bool = True
    min_snr_gamma: float = 5.0

    # Classifier-free guidance
    cfg_prob: float = 0.15
    cfg_scale: float = 1.5
    use_diversity_preserving_cfg: bool = True

    # GMM Subtypes
    max_subtypes: int = 4
    min_samples_per_subtype: int = 80
    gmm_covariance_type: str = "diag"
    gmm_reg_covar: float = 1e-4
    subtype_conditioning_mode: str = "hard"  # {"none", "hard", "random_within_class", "global_hard"}

    # Sampling
    balanced_sampling: bool = True

    # EMA (still disabled for small datasets)
    use_ema: bool = False
    ema_decay: float = 0.999

    # Calibration
    use_calibration: bool = True
    calibration_clip: float = 6.0

    # Logging
    print_every: int = 20

    # Prior-based forward noising: relevant dims destroyed slowly
    use_prior_slow_noising: bool = True
    corruption_law: str = "adaptive"   # {"adaptive", "isotropic", "random_protected", "dense_class_agnostic"}
    prior_kappa: float = 0.35        # strength of protection
    prior_power: float = 2.0         # anneal curvature
    prior_mask_temperature: float = 1.0  # softness for mask from |d|
    prior_mask_momentum: float = 0.9     # EMA for stability (optional)


# =============================================================================
# Prior and Sampler
# =============================================================================

class GDTPrior:
    """Class-conditional diffusion prior on correlation-manifold latents."""

    def __init__(
        self,
        device: str = "cpu",
        config: Optional[GDTConfig] = None,
        random_state: Optional[int] = None,
    ):
        self.device = torch.device(device)
        self.cfg = config or GDTConfig()
        if isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            self.rng = np.random.RandomState(random_state)

        self.model_: Optional[nn.Module] = None
        self.ema_model_: Optional[nn.Module] = None
        self.adapter_: Optional[CorrCholeskyAdapter] = None
        self.alpha_bars_: Optional[torch.Tensor] = None
        self._fitted = False

        if self.cfg.use_diversity_preserving_cfg:
            self.div_cfg = DiversityPreservingCFG(
                base_scale=self.cfg.cfg_scale,
                diversity_weight=0.3,
            )
        else:
            self.div_cfg = None

        self._latent_calibrator: Optional[Dict[int, Dict[str, np.ndarray]]] = None
        self.class_subtype_offsets_: Dict[int, int] = {}
        self.class_subtype_counts_: Dict[int, int] = {}
        self.class_subtype_probs_: Dict[int, np.ndarray] = {}
        self.class_subtype_gmm_params_: Dict[int, Dict[str, np.ndarray]] = {}
        self._gmm_tensor_cache_: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
        self.subtype_fit_diagnostics_: Dict[str, object] = {}
        self._train_cond_idx: Optional[np.ndarray] = None
        self.structural_atom_params_: Dict[int, Dict[str, np.ndarray]] = {}
        self._structural_atom_tensor_cache_: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}

    def _init_schedule(self):
        """Initialize noise schedule."""
        T = self.cfg.diffusion_steps
        if self.cfg.schedule == "cosine":
            s = 0.008
            t = torch.arange(T + 1, device=self.device, dtype=torch.float32)
            f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bars = (f[:-1] / f[0]).clamp(1e-8, 1.0)
        else:
            betas = torch.linspace(1e-4, 0.02, T, device=self.device)
            alpha_bars = torch.cumprod(1 - betas, dim=0).clamp(1e-8, 1.0)
        self.alpha_bars_ = alpha_bars

    def _init_ema(self):
        if self.cfg.use_ema:
            self.ema_model_ = copy.deepcopy(self.model_).eval()
            for p in self.ema_model_.parameters():
                p.requires_grad_(False)

    def _update_ema(self):
        if self.ema_model_ is None:
            return
        d = self.cfg.ema_decay
        with torch.no_grad():
            for p_ema, p in zip(self.ema_model_.parameters(), self.model_.parameters()):
                p_ema.data.mul_(d).add_(p.data, alpha=1 - d)

    def _bridge_like_mode_enabled(self) -> bool:
        return self.cfg.use_spectral_cond and self.cfg.spectral_feature_mode in {
            "amortized_bridge_v1",
            "teacher_bridge_v1",
        }

    def _teacher_bridge_mode_enabled(self) -> bool:
        return self.cfg.use_spectral_cond and self.cfg.spectral_feature_mode == "teacher_bridge_v1"

    def _build_sample_weights(self, subject_ids: Optional[np.ndarray], n_samples: int) -> np.ndarray:
        if subject_ids is None:
            return np.ones(n_samples, dtype=np.float32)
        s = np.asarray(subject_ids)
        _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
        w = 1.0 / counts[inv].astype(np.float64)
        w = w / np.mean(w)
        return w.astype(np.float32)

    def _fit_latent_calibration(self, z_real: np.ndarray, z_prior: np.ndarray, y_idx: np.ndarray) -> None:
        z_real = np.asarray(z_real, dtype=np.float64)
        z_prior = np.asarray(z_prior, dtype=np.float64)
        y_idx = np.asarray(y_idx, dtype=np.int64)

        calibrator = {}
        eps = 1e-6

        for cls in range(self.n_classes_):
            mask = y_idx == cls
            if not np.any(mask):
                continue

            zr, zp = z_real[mask], z_prior[mask]
            mean_r, std_r = zr.mean(axis=0), zr.std(axis=0)
            mean_p, std_p = zp.mean(axis=0), zp.std(axis=0)

            scale = std_r / (std_p + eps)
            scale = np.clip(scale, 0.5, 2.0)
            shift = mean_r - scale * mean_p

            calibrator[int(cls)] = {"scale": scale.astype(np.float64), "shift": shift.astype(np.float64)}

        self._latent_calibrator = calibrator

    def _apply_latent_calibration(self, z: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
        if self._latent_calibrator is None:
            return np.asarray(z, dtype=np.float64)

        out = np.asarray(z, dtype=np.float64).copy()
        y_idx = np.asarray(y_idx, dtype=np.int64)
        clip_val = float(self.cfg.calibration_clip)

        for cls, params in self._latent_calibrator.items():
            mask = y_idx == cls
            if np.any(mask):
                out[mask] = out[mask] * params["scale"] + params["shift"]

        return np.clip(out, -clip_val, clip_val)

    def _fit_subtype_tokens(self, X: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
        n = len(y_idx)
        token_idx = np.zeros(n, dtype=np.int64)

        self.class_subtype_offsets_ = {}
        self.class_subtype_counts_ = {}
        self.class_subtype_probs_ = {}
        self.class_subtype_gmm_params_ = {}
        self._gmm_tensor_cache_ = {}
        bic_curves: Dict[int, Dict[str, object]] = {}

        offset = 0
        for cls in range(self.n_classes_):
            idx = np.where(y_idx == cls)[0]
            Xc = X[idx]
            n_c = len(Xc)
            bic_entries: list[tuple[int, float]] = []

            if n_c <= 2:
                best_k, best_model = 1, None
            else:
                max_k_data = max(1, n_c // max(1, self.cfg.min_samples_per_subtype))
                max_k = min(self.cfg.max_subtypes, max_k_data, n_c - 1)
                best_k, best_model, best_bic = 1, None, np.inf

                for k in range(1, max_k + 1):
                    try:
                        gm = GaussianMixture(
                            n_components=k,
                            covariance_type=self.cfg.gmm_covariance_type,
                            reg_covar=self.cfg.gmm_reg_covar,
                            random_state=int(self.rng.randint(0, 2**31 - 1)),
                        )
                        gm.fit(Xc)
                        bic = gm.bic(Xc)
                        bic_entries.append((int(k), float(bic)))
                        if bic < best_bic:
                            best_bic, best_k, best_model = bic, k, gm
                    except:
                        continue

            self.class_subtype_offsets_[cls] = offset
            self.class_subtype_counts_[cls] = best_k

            if best_k == 1 or best_model is None:
                local = np.zeros(n_c, dtype=np.int64)
                probs = np.array([1.0])
                means = Xc.mean(axis=0, keepdims=True).astype(np.float32)
                variances = np.clip(
                    Xc.var(axis=0, keepdims=True).astype(np.float32) + float(self.cfg.gmm_reg_covar),
                    1e-6,
                    None,
                )
            else:
                local = best_model.predict(Xc).astype(np.int64)
                probs = best_model.weights_.astype(np.float64)
                probs = probs / probs.sum()
                means = best_model.means_.astype(np.float32)
                if best_model.covariance_type == "diag":
                    variances = np.asarray(best_model.covariances_, dtype=np.float32)
                elif best_model.covariance_type == "full":
                    variances = np.diagonal(
                        np.asarray(best_model.covariances_, dtype=np.float32),
                        axis1=1,
                        axis2=2,
                    )
                elif best_model.covariance_type == "spherical":
                    spherical = np.asarray(best_model.covariances_, dtype=np.float32)
                    variances = np.repeat(spherical[:, None], Xc.shape[1], axis=1)
                else:  # tied
                    tied_diag = np.diagonal(np.asarray(best_model.covariances_, dtype=np.float32))
                    variances = np.repeat(tied_diag[None, :], best_k, axis=0)
                variances = np.clip(variances, 1e-6, None).astype(np.float32)

            token_idx[idx] = offset + local
            self.class_subtype_probs_[cls] = probs
            self.class_subtype_gmm_params_[cls] = {
                "weights": probs.astype(np.float32),
                "means": means.astype(np.float32),
                "variances": variances.astype(np.float32),
            }
            bic_curves[int(cls)] = {
                "n_samples": int(n_c),
                "k": [int(k) for k, _ in bic_entries],
                "bic": [float(bic) for _, bic in bic_entries],
            }
            offset += best_k

        self.n_conditions_ = int(offset)
        self.uncond_idx_ = self.n_conditions_
        self.subtype_fit_diagnostics_ = {
            "fit_type": "classwise_gmm",
            "class_bic_curves": bic_curves,
        }
        return token_idx

    def _fit_class_only_tokens(self, X: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
        n = len(y_idx)
        token_idx = np.zeros(n, dtype=np.int64)

        self.class_subtype_offsets_ = {}
        self.class_subtype_counts_ = {}
        self.class_subtype_probs_ = {}
        self.class_subtype_gmm_params_ = {}
        self._gmm_tensor_cache_ = {}

        offset = 0
        for cls in range(self.n_classes_):
            idx = np.where(y_idx == cls)[0]
            Xc = X[idx]
            token_idx[idx] = offset
            self.class_subtype_offsets_[cls] = offset
            self.class_subtype_counts_[cls] = 1
            self.class_subtype_probs_[cls] = np.array([1.0], dtype=np.float64)
            means = Xc.mean(axis=0, keepdims=True).astype(np.float32)
            variances = np.clip(
                Xc.var(axis=0, keepdims=True).astype(np.float32) + float(self.cfg.gmm_reg_covar),
                1e-6,
                None,
            )
            self.class_subtype_gmm_params_[cls] = {
                "weights": np.array([1.0], dtype=np.float32),
                "means": means,
                "variances": variances,
            }
            offset += 1

        self.n_conditions_ = int(offset)
        self.uncond_idx_ = self.n_conditions_
        self.subtype_fit_diagnostics_ = {
            "fit_type": "class_only",
            "class_bic_curves": {},
        }
        return token_idx

    def _fit_random_within_class_tokens(self, X: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
        cond_idx = self._fit_subtype_tokens(X, y_idx)
        randomized = cond_idx.copy()
        randomized_counts: Dict[int, list[int]] = {}

        for cls in range(self.n_classes_):
            idx = np.where(y_idx == cls)[0]
            offset = int(self.class_subtype_offsets_[cls])
            k = int(self.class_subtype_counts_[cls])
            local = cond_idx[idx] - offset
            counts = np.bincount(local, minlength=k).astype(np.int64)
            labels = np.concatenate(
                [
                    np.full(int(count), offset + local_id, dtype=np.int64)
                    for local_id, count in enumerate(counts)
                ]
            )
            if labels.size:
                labels = labels[self.rng.permutation(labels.size)]
                randomized[idx] = labels
            randomized_counts[int(cls)] = counts.tolist()
            probs = counts.astype(np.float64)
            probs = probs / max(probs.sum(), 1.0)
            self.class_subtype_probs_[cls] = probs
            self.class_subtype_gmm_params_[cls]["weights"] = probs.astype(np.float32)

        self.subtype_fit_diagnostics_["fit_type"] = "random_within_class"
        self.subtype_fit_diagnostics_["randomized_counts"] = randomized_counts
        return randomized

    def _fit_global_subtype_tokens(self, X: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
        n = len(y_idx)
        token_idx = np.zeros(n, dtype=np.int64)

        self.class_subtype_offsets_ = {}
        self.class_subtype_counts_ = {}
        self.class_subtype_probs_ = {}
        self.class_subtype_gmm_params_ = {}
        self._gmm_tensor_cache_ = {}

        bic_entries: list[tuple[int, float]] = []
        if n <= 2:
            best_k, best_model = 1, None
        else:
            max_k_data = max(1, n // max(1, self.cfg.min_samples_per_subtype))
            max_k = min(self.cfg.max_subtypes, max_k_data, n - 1)
            best_k, best_model, best_bic = 1, None, np.inf
            for k in range(1, max_k + 1):
                try:
                    gm = GaussianMixture(
                        n_components=k,
                        covariance_type=self.cfg.gmm_covariance_type,
                        reg_covar=self.cfg.gmm_reg_covar,
                        random_state=int(self.rng.randint(0, 2**31 - 1)),
                    )
                    gm.fit(X)
                    bic = gm.bic(X)
                    bic_entries.append((int(k), float(bic)))
                    if bic < best_bic:
                        best_bic, best_k, best_model = bic, k, gm
                except:
                    continue

        if best_k == 1 or best_model is None:
            global_local = np.zeros(n, dtype=np.int64)
            global_means = X.mean(axis=0, keepdims=True).astype(np.float32)
            global_variances = np.clip(
                X.var(axis=0, keepdims=True).astype(np.float32) + float(self.cfg.gmm_reg_covar),
                1e-6,
                None,
            )
        else:
            global_local = best_model.predict(X).astype(np.int64)
            global_means = best_model.means_.astype(np.float32)
            if best_model.covariance_type == "diag":
                global_variances = np.asarray(best_model.covariances_, dtype=np.float32)
            elif best_model.covariance_type == "full":
                global_variances = np.diagonal(
                    np.asarray(best_model.covariances_, dtype=np.float32),
                    axis1=1,
                    axis2=2,
                )
            elif best_model.covariance_type == "spherical":
                spherical = np.asarray(best_model.covariances_, dtype=np.float32)
                global_variances = np.repeat(spherical[:, None], X.shape[1], axis=1)
            else:
                tied_diag = np.diagonal(np.asarray(best_model.covariances_, dtype=np.float32))
                global_variances = np.repeat(tied_diag[None, :], best_k, axis=0)
            global_variances = np.clip(global_variances, 1e-6, None).astype(np.float32)

        offset = 0
        for cls in range(self.n_classes_):
            idx = np.where(y_idx == cls)[0]
            local_counts = np.bincount(global_local[idx], minlength=best_k).astype(np.float64)
            probs = local_counts / max(local_counts.sum(), 1.0)
            self.class_subtype_offsets_[cls] = offset
            self.class_subtype_counts_[cls] = int(best_k)
            self.class_subtype_probs_[cls] = probs
            self.class_subtype_gmm_params_[cls] = {
                "weights": probs.astype(np.float32),
                "means": global_means.astype(np.float32),
                "variances": global_variances.astype(np.float32),
            }
            token_idx[idx] = offset + global_local[idx]
            offset += int(best_k)

        self.n_conditions_ = int(offset)
        self.uncond_idx_ = self.n_conditions_
        self.subtype_fit_diagnostics_ = {
            "fit_type": "global_gmm",
            "global_bic_curve": {
                "n_samples": int(n),
                "k": [int(k) for k, _ in bic_entries],
                "bic": [float(bic) for _, bic in bic_entries],
            },
            "global_k": int(best_k),
        }
        return token_idx

    def _no_subtype_mode_enabled(self) -> bool:
        return self.cfg.subtype_conditioning_mode == "none"

    def _random_within_class_mode_enabled(self) -> bool:
        return self.cfg.subtype_conditioning_mode == "random_within_class"

    def _global_hard_mode_enabled(self) -> bool:
        return self.cfg.subtype_conditioning_mode == "global_hard"

    def _make_unconditional_condition(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size,), self.uncond_idx_, device=device, dtype=torch.long)

    def _fit_structural_atoms(self, X: np.ndarray, cond_idx: np.ndarray) -> None:
        """Fit subtype-specific structural atoms in proxy_v1 feature space."""
        self.structural_atom_params_ = {}
        self._structural_atom_tensor_cache_ = {}
        if not self._bridge_like_mode_enabled():
            return

        x_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        feature_batches = []
        chunk_size = max(64, min(512, self.cfg.batch_size * 4))
        with torch.no_grad():
            for start in range(0, x_tensor.shape[0], chunk_size):
                stop = min(start + chunk_size, x_tensor.shape[0])
                feature_batches.append(
                    self.model_.spectral_module.compute_spectral_features(x_tensor[start:stop]).cpu()
                )
        spectral_features = torch.cat(feature_batches, dim=0).numpy().astype(np.float32)
        min_side = max(1, self.cfg.min_samples_per_subtype // 4)

        for token in range(self.n_conditions_):
            idx = np.where(cond_idx == token)[0]
            if idx.size == 0:
                continue
            feats = spectral_features[idx]
            scale = np.clip(feats.var(axis=0) + float(self.cfg.gmm_reg_covar), 1e-6, None).astype(np.float32)

            atoms = feats.mean(axis=0, keepdims=True).astype(np.float32)
            weights = np.array([1.0], dtype=np.float32)

            if idx.size >= 2 * self.cfg.min_samples_per_subtype:
                centered = feats - feats.mean(axis=0, keepdims=True)
                try:
                    _, _, vh = np.linalg.svd(centered, full_matrices=False)
                    direction = vh[0]
                except np.linalg.LinAlgError:
                    direction = None

                if direction is not None and np.linalg.norm(direction) > 0:
                    proj = centered @ direction.astype(np.float32)
                    threshold = float(np.median(proj))
                    left = proj <= threshold
                    right = proj > threshold
                    if int(left.sum()) >= min_side and int(right.sum()) >= min_side:
                        atoms = np.stack(
                            [feats[left].mean(axis=0), feats[right].mean(axis=0)],
                            axis=0,
                        ).astype(np.float32)
                        weights = np.array(
                            [float(left.mean()), float(right.mean())],
                            dtype=np.float32,
                        )
                        weights = weights / max(weights.sum(), 1e-8)

            self.structural_atom_params_[int(token)] = {
                "atoms": atoms.astype(np.float32),
                "weights": weights.astype(np.float32),
                "scales": scale.astype(np.float32),
            }

    def _get_structural_atom_tensors(self, x_like: torch.Tensor) -> Dict[int, Dict[str, torch.Tensor]]:
        cache_key = f"{x_like.device}_{str(x_like.dtype)}"
        if cache_key in self._structural_atom_tensor_cache_:
            return self._structural_atom_tensor_cache_[cache_key]

        cache: Dict[int, Dict[str, torch.Tensor]] = {}
        for token, params in self.structural_atom_params_.items():
            atoms = torch.from_numpy(params["atoms"]).to(device=x_like.device, dtype=x_like.dtype)
            weights = torch.from_numpy(params["weights"]).to(device=x_like.device, dtype=x_like.dtype).clamp_min(1e-8)
            weights = weights / weights.sum()
            scales = torch.from_numpy(params["scales"]).to(device=x_like.device, dtype=x_like.dtype).clamp_min(1e-6)
            cache[int(token)] = {
                "atoms": atoms,
                "log_prior": torch.log(weights),
                "scales": scales,
            }

        self._structural_atom_tensor_cache_[cache_key] = cache
        return cache

    def _bridge_token_probabilities(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one-hot subtype-token probabilities excluding the unconditional token."""
        probs = torch.zeros(cond.shape[0], self.n_conditions_, device=cond.device, dtype=torch.float32)
        valid_mask = cond != self.uncond_idx_
        if torch.any(valid_mask):
            probs[valid_mask, cond[valid_mask]] = 1.0
        return probs, valid_mask

    def _normalize_alpha_bar(
        self,
        alpha_bar: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if alpha_bar.ndim == 0:
            alpha_bar = alpha_bar.view(1, 1)
        elif alpha_bar.ndim == 1:
            alpha_bar = alpha_bar.unsqueeze(-1)
        alpha_bar = alpha_bar.to(device=device, dtype=dtype)
        if alpha_bar.shape[0] == 1 and batch_size != 1:
            alpha_bar = alpha_bar.expand(batch_size, -1)
        return alpha_bar

    def _build_mixture_bridge_conditioning(
        self,
        x_ref: torch.Tensor,
        cond: torch.Tensor,
        alpha_bar: torch.Tensor,
        prev_bridge_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Subtype-conditioned entropy-annealed structural mixture bridge."""
        batch_size = x_ref.shape[0]
        alpha_bar = self._normalize_alpha_bar(alpha_bar, batch_size, x_ref.device, x_ref.dtype)
        spectral_cond = self.model_.spectral_module(None, batch_size=batch_size, device=x_ref.device)
        bridge_features = x_ref.new_zeros((batch_size, self.model_.spectral_module.n_spectral))
        token_probs, valid_mask = self._bridge_token_probabilities(cond)
        if not torch.any(valid_mask):
            return spectral_cond, bridge_features

        cache = self._get_structural_atom_tensors(x_ref)
        query = self.model_.spectral_module.compute_spectral_features(x_ref[valid_mask])
        token_probs_valid = token_probs[valid_mask]
        alpha_valid = alpha_bar[valid_mask]
        entropy_temp = (1.0 - alpha_valid).clamp(min=0.05, max=1.0)
        temporal_weight = alpha_valid
        prev_valid = None if prev_bridge_features is None else prev_bridge_features[valid_mask]

        bary_features = x_ref.new_zeros((query.shape[0], self.model_.spectral_module.n_spectral))
        active_token_ids = torch.nonzero(token_probs_valid.sum(dim=0) > 1e-8, as_tuple=False).flatten()
        for token in active_token_ids:
            token_id = int(token.item())
            params = cache[token_id]
            token_weight = token_probs_valid[:, token_id]
            token_mask = token_weight > 1e-8
            if not torch.any(token_mask):
                continue
            q = query[token_mask]
            atoms = params["atoms"]
            scales = params["scales"]
            logits = params["log_prior"].unsqueeze(0).expand(q.shape[0], -1)

            diff = q.unsqueeze(1) - atoms.unsqueeze(0)
            cost = torch.mean((diff * diff) / scales.unsqueeze(0), dim=-1)
            logits = logits - cost

            if prev_valid is not None:
                prev_bary = prev_valid[token_mask]
                trans = atoms.unsqueeze(0) - prev_bary.unsqueeze(1)
                trans_cost = torch.mean((trans * trans) / scales.unsqueeze(0), dim=-1)
                logits = logits - temporal_weight[token_mask] * trans_cost

            weights = torch.softmax(logits / entropy_temp[token_mask], dim=-1)
            bary_features[token_mask] = bary_features[token_mask] + token_weight[token_mask].unsqueeze(-1) * (weights @ atoms)

        spectral_cond = spectral_cond.clone()
        spectral_cond[valid_mask] = self.model_.spectral_module.embed_features(bary_features)
        bridge_features[valid_mask] = bary_features
        return spectral_cond, bridge_features

    def _amortized_bridge_mode_enabled(self) -> bool:
        return self.cfg.use_spectral_cond and self.cfg.spectral_feature_mode == "amortized_bridge_v1"

    def _build_amortized_bridge_conditioning(
        self,
        x_t: torch.Tensor,
        cond: torch.Tensor,
        t_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cheap amortized approximation to mixture-bridge conditioning."""
        batch_size = x_t.shape[0]
        spectral_cond = self.model_.spectral_module(None, batch_size=batch_size, device=x_t.device)
        bridge_features = x_t.new_zeros((batch_size, self.model_.spectral_module.n_spectral))

        _, valid_mask = self._bridge_token_probabilities(cond)
        if not torch.any(valid_mask):
            return spectral_cond, bridge_features

        pred_features = self.model_.predict_amortized_bridge_features(
            x_t[valid_mask],
            cond[valid_mask],
            t_norm[valid_mask],
        )
        spectral_cond = spectral_cond.clone()
        spectral_cond[valid_mask] = self.model_.spectral_module.embed_features(pred_features)
        bridge_features[valid_mask] = pred_features
        return spectral_cond, bridge_features

    def _compute_amortized_bridge_loss(
        self,
        bridge_pred: torch.Tensor,
        bridge_target: torch.Tensor,
        cond: torch.Tensor,
        alpha_bar: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        _, valid_mask = self._bridge_token_probabilities(cond)
        if not torch.any(valid_mask):
            return bridge_pred.new_tensor(0.0)

        per_sample = torch.mean((bridge_pred - bridge_target) ** 2, dim=-1)
        alpha = self._normalize_alpha_bar(alpha_bar, bridge_pred.shape[0], bridge_pred.device, bridge_pred.dtype).squeeze(-1)
        combined_weight = alpha * sample_weights
        combined_weight = combined_weight * valid_mask.to(combined_weight.dtype)
        weighted = per_sample * combined_weight
        return weighted.sum() / (combined_weight.sum() + 1e-8)

    def _sample_condition_tokens(self, y_cond: np.ndarray) -> np.ndarray:
        class_idx = np.array([self.class_map_[int(c)] for c in y_cond], dtype=np.int64)
        out = np.empty_like(class_idx)
        for i, cls in enumerate(class_idx):
            probs = self.class_subtype_probs_[int(cls)]
            local = int(self.rng.choice(len(probs), p=probs))
            out[i] = self.class_subtype_offsets_[int(cls)] + local
        return out

    def export_subtype_artifacts(self) -> Dict[str, np.ndarray]:
        if self._train_cond_idx is None or self.adapter_ is None:
            return {}

        token_class_idx = []
        token_local_idx = []
        token_weights = []
        token_means = []
        for cls in range(self.n_classes_):
            k = int(self.class_subtype_counts_[cls])
            probs = np.asarray(self.class_subtype_probs_[cls], dtype=np.float32)
            means = np.asarray(self.class_subtype_gmm_params_[cls]["means"], dtype=np.float32)
            for local in range(k):
                token_class_idx.append(int(cls))
                token_local_idx.append(int(local))
                token_weights.append(float(probs[min(local, len(probs) - 1)]))
                token_means.append(means[min(local, means.shape[0] - 1)])

        token_means_arr = np.asarray(token_means, dtype=np.float32)
        token_corr_centroids = self.adapter_.x_to_corr_np(token_means_arr).astype(np.float32)
        train_token_counts = np.bincount(self._train_cond_idx, minlength=self.n_conditions_).astype(np.int64)
        train_token_probs = train_token_counts.astype(np.float64)
        train_token_probs = train_token_probs / max(train_token_probs.sum(), 1.0)

        return {
            "classes": np.asarray(self.classes_, dtype=np.int64),
            "train_y_idx": np.asarray(self._y_idx_train, dtype=np.int64),
            "train_cond_idx": np.asarray(self._train_cond_idx, dtype=np.int64),
            "token_class_idx": np.asarray(token_class_idx, dtype=np.int64),
            "token_local_idx": np.asarray(token_local_idx, dtype=np.int64),
            "token_weights": np.asarray(token_weights, dtype=np.float32),
            "token_corr_centroids": token_corr_centroids,
            "train_token_counts": train_token_counts,
            "train_token_probs": train_token_probs.astype(np.float32),
            "z_mean": np.asarray(self.adapter_.z_mean, dtype=np.float32),
            "z_scale": np.asarray(self.adapter_.z_scale, dtype=np.float32),
        }

    def export_subtype_summary(self) -> Dict[str, object]:
        def _to_serializable(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.integer, np.floating)):
                return value.item()
            if isinstance(value, dict):
                return {str(k): _to_serializable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_to_serializable(v) for v in value]
            return value

        train_token_counts = []
        if self._train_cond_idx is not None:
            train_token_counts = np.bincount(self._train_cond_idx, minlength=self.n_conditions_).astype(int).tolist()

        return {
            "mode": self.cfg.subtype_conditioning_mode,
            "max_subtypes": int(self.cfg.max_subtypes),
            "min_samples_per_subtype": int(self.cfg.min_samples_per_subtype),
            "classes": _to_serializable(np.asarray(self.classes_, dtype=np.int64)),
            "class_map": {str(k): int(v) for k, v in self.class_map_.items()},
            "class_subtype_offsets": {str(k): int(v) for k, v in self.class_subtype_offsets_.items()},
            "class_subtype_counts": {str(k): int(v) for k, v in self.class_subtype_counts_.items()},
            "class_subtype_probs": {str(k): _to_serializable(v) for k, v in self.class_subtype_probs_.items()},
            "n_conditions": int(self.n_conditions_),
            "uncond_idx": int(self.uncond_idx_),
            "train_token_counts": train_token_counts,
            "fit_diagnostics": _to_serializable(self.subtype_fit_diagnostics_),
        }

    def infer_condition_tokens(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Infer hard condition-token assignments for arbitrary latent samples."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        y_idx = np.array([self.class_map_[int(c)] for c in y], dtype=np.int64)
        token_idx = np.empty(len(y_idx), dtype=np.int64)

        for cls in range(self.n_classes_):
            idx = np.where(y_idx == cls)[0]
            if idx.size == 0:
                continue
            offset = int(self.class_subtype_offsets_[cls])
            params = self.class_subtype_gmm_params_[cls]
            weights = np.asarray(params["weights"], dtype=np.float64)
            means = np.asarray(params["means"], dtype=np.float64)
            variances = np.clip(np.asarray(params["variances"], dtype=np.float64), 1e-6, None)
            k = int(means.shape[0])
            if k == 1:
                token_idx[idx] = offset
                continue

            Xc = np.asarray(X[idx], dtype=np.float64)
            diff = Xc[:, None, :] - means[None, :, :]
            log_det = np.log(variances).sum(axis=1, keepdims=True).T
            quad = (diff * diff / variances[None, :, :]).sum(axis=2)
            log_prob = np.log(np.clip(weights, 1e-8, None))[None, :] - 0.5 * (log_det + quad)
            local = np.argmax(log_prob, axis=1).astype(np.int64)
            token_idx[idx] = offset + local

        return token_idx

    def export_probe_bundle(self) -> Dict[str, object]:
        """Export the fitted bridge state required for post-hoc timestep probing."""
        if not self._fitted or self.model_ is None or self.adapter_ is None:
            raise RuntimeError("Bridge probe export requires a fitted prior.")

        model = self._get_model()
        model_state = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
        return {
            "cfg": asdict(self.cfg),
            "classes": np.asarray(self.classes_, dtype=np.int64),
            "class_subtype_offsets": {int(k): int(v) for k, v in self.class_subtype_offsets_.items()},
            "class_subtype_counts": {int(k): int(v) for k, v in self.class_subtype_counts_.items()},
            "class_subtype_probs": {
                int(k): np.asarray(v, dtype=np.float32)
                for k, v in self.class_subtype_probs_.items()
            },
            "class_subtype_gmm_params": {
                int(k): {
                    "weights": np.asarray(v["weights"], dtype=np.float32),
                    "means": np.asarray(v["means"], dtype=np.float32),
                    "variances": np.asarray(v["variances"], dtype=np.float32),
                }
                for k, v in self.class_subtype_gmm_params_.items()
            },
            "structural_atom_params": {
                int(k): {
                    "atoms": np.asarray(v["atoms"], dtype=np.float32),
                    "weights": np.asarray(v["weights"], dtype=np.float32),
                    "scales": np.asarray(v["scales"], dtype=np.float32),
                }
                for k, v in self.structural_atom_params_.items()
            },
            "adapter": {
                "n_features": int(self.adapter_.n_features),
                "z_mean": np.asarray(self.adapter_.z_mean, dtype=np.float32),
                "z_scale": np.asarray(self.adapter_.z_scale, dtype=np.float32),
            },
            "n_conditions": int(self.n_conditions_),
            "uncond_idx": int(self.uncond_idx_),
            "latent_calibrator": self._latent_calibrator,
            "model_state_dict": model_state,
        }

    @classmethod
    def from_probe_bundle(
        cls,
        payload: Dict[str, object],
        device: str = "cpu",
    ) -> "GDTPrior":
        """Reconstruct a fitted prior from a saved bridge-probe bundle."""
        cfg = GDTConfig(**payload["cfg"])
        prior = cls(device=device, config=cfg, random_state=0)

        adapter_payload = payload["adapter"]
        prior.adapter_ = CorrCholeskyAdapter(
            int(adapter_payload["n_features"]),
            z_mean=np.asarray(adapter_payload["z_mean"], dtype=np.float32),
            z_scale=np.asarray(adapter_payload["z_scale"], dtype=np.float32),
        )

        prior.classes_ = np.asarray(payload["classes"], dtype=np.int64)
        prior.n_classes_ = int(len(prior.classes_))
        prior.class_map_ = {int(c): i for i, c in enumerate(prior.classes_)}
        prior.class_subtype_offsets_ = {
            int(k): int(v)
            for k, v in payload["class_subtype_offsets"].items()
        }
        prior.class_subtype_counts_ = {
            int(k): int(v)
            for k, v in payload["class_subtype_counts"].items()
        }
        prior.class_subtype_probs_ = {
            int(k): np.asarray(v, dtype=np.float64)
            for k, v in payload["class_subtype_probs"].items()
        }
        prior.class_subtype_gmm_params_ = {
            int(k): {
                "weights": np.asarray(v["weights"], dtype=np.float32),
                "means": np.asarray(v["means"], dtype=np.float32),
                "variances": np.asarray(v["variances"], dtype=np.float32),
            }
            for k, v in payload["class_subtype_gmm_params"].items()
        }
        prior.structural_atom_params_ = {
            int(k): {
                "atoms": np.asarray(v["atoms"], dtype=np.float32),
                "weights": np.asarray(v["weights"], dtype=np.float32),
                "scales": np.asarray(v["scales"], dtype=np.float32),
            }
            for k, v in payload.get("structural_atom_params", {}).items()
        }
        prior.n_conditions_ = int(payload["n_conditions"])
        prior.uncond_idx_ = int(payload["uncond_idx"])
        prior._latent_calibrator = payload.get("latent_calibrator")

        prior.model_ = _build_denoiser(
            prior.cfg,
            n_nodes=prior.adapter_.n_nodes,
            n_conditions=prior.n_conditions_ + 1,
        ).to(prior.device)
        prior.model_.load_state_dict(payload["model_state_dict"])
        prior.model_.eval()
        for param in prior.model_.parameters():
            param.requires_grad_(False)

        prior._init_schedule()
        prior._fitted = True
        return prior

    def _get_balanced_indices(self, y_idx: np.ndarray, batch_size: int) -> np.ndarray:
        classes = np.unique(y_idx)
        n_per_class = batch_size // len(classes)
        remainder = batch_size % len(classes)

        indices = []
        for i, c in enumerate(classes):
            class_indices = np.where(y_idx == c)[0]
            n = n_per_class + (1 if i < remainder else 0)
            indices.append(self.rng.choice(class_indices, size=n, replace=True))

        indices = np.concatenate(indices)
        self.rng.shuffle(indices)
        return indices

    # Noise adaptation based on priors
    def _tau(self, t_idx: torch.Tensor) -> torch.Tensor:
        return t_idx.float() / max(self.cfg.diffusion_steps - 1, 1)

    def _prior_strength(self, t_idx: torch.Tensor) -> torch.Tensor:
        # strong early, decays to 0 late
        tau = self._tau(t_idx)
        return (1.0 - tau).pow(self.cfg.prior_power).unsqueeze(-1)  # [B,1]

    def _masked_topk_relevance(self, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Keep only the top-q coordinates, where q is induced by the time schedule."""
        M = m.size(1)
        q = (1.0 - s).clamp(0.05, 1.0).squeeze(-1)
        k_keep = torch.ceil(q * M).long().clamp(1, M)
        k_max = int(k_keep.max().item())
        top_vals, _ = torch.topk(m, k_max, dim=1, largest=True, sorted=True)
        thr = top_vals.gather(1, (k_keep - 1).clamp(0, k_max - 1).unsqueeze(1))
        return m * (m >= thr).to(m.dtype)

    def _random_protected_mask(
        self,
        batch_size: int,
        n_features: int,
        s: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Protect the same fraction of coordinates as adaptive corruption, but choose them uniformly at random."""
        q = (1.0 - s).clamp(0.05, 1.0).squeeze(-1)
        k_keep = torch.ceil(q * n_features).long().clamp(1, n_features)
        rand = torch.rand(batch_size, n_features, device=device, dtype=dtype)
        k_max = int(k_keep.max().item())
        top_vals, _ = torch.topk(rand, k_max, dim=1, largest=True, sorted=True)
        thr = top_vals.gather(1, (k_keep - 1).clamp(0, k_max - 1).unsqueeze(1))
        return (rand >= thr).to(dtype)

    @torch.no_grad()
    def _apply_corruption_law(
        self,
        x0: torch.Tensor,
        ab: torch.Tensor,
        t_idx: torch.Tensor,
        t_norm: torch.Tensor,
        yb: torch.Tensor,
        cb_hard: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Apply the configured forward corruption law and return (x_t, noise_eff, m_anchor, s_anchor)."""
        noise = torch.randn_like(x0)
        if not self.cfg.use_prior_slow_noising or self.cfg.prior_kappa <= 0:
            return (
                torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise,
                noise,
                None,
                None,
            )

        law = str(self.cfg.corruption_law)
        if law == "isotropic":
            return (
                torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise,
                noise,
                None,
                None,
            )

        s = self._prior_strength(t_idx)
        kappa = float(self.cfg.prior_kappa)

        if law == "adaptive":
            noise0 = torch.randn_like(x0)
            x_t0 = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise0
            m = self._guidance_relevance_mask(
                x_t=x_t0,
                t_norm=t_norm,
                y_class_idx=yb,
                hard_cond=cb_hard,
                spectral_cond=None,
            )
            m = self._masked_topk_relevance(m, s)
            sigma = (1.0 - kappa * s * m).clamp(min=0.05, max=1.0)
            noise_eff = sigma * noise
            x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise_eff
            return x_t, noise_eff, m.detach(), s.detach()

        if law == "random_protected":
            m = self._random_protected_mask(
                batch_size=x0.shape[0],
                n_features=x0.shape[1],
                s=s,
                device=x0.device,
                dtype=x0.dtype,
            )
            sigma = (1.0 - kappa * s * m).clamp(min=0.05, max=1.0)
            noise_eff = sigma * noise
            x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise_eff
            return x_t, noise_eff, m.detach(), s.detach()

        if law == "dense_class_agnostic":
            sigma = (1.0 - kappa * s).clamp(min=0.05, max=1.0)
            noise_eff = sigma * noise
            x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise_eff
            return x_t, noise_eff, None, s.detach()

        raise ValueError(
            "Unsupported corruption_law: "
            f"{law!r}. Expected one of ['adaptive', 'isotropic', 'random_protected', 'dense_class_agnostic']."
        )

    @torch.no_grad()
    def _guidance_relevance_mask(
        self,
        x_t: torch.Tensor,
        t_norm: torch.Tensor,
        y_class_idx: torch.Tensor,
        hard_cond: Optional[torch.Tensor],
        spectral_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns m in [0,1]^M: per-dimension relevance mask from guidance signal.
        """
        model = self._get_model()
        model.eval()

        if hard_cond is None:
            raise RuntimeError("hard_cond must be provided for guidance relevance masking.")
        cond_c = hard_cond

        pred_c = model(x_t, cond_c, t_norm, spectral_cond=spectral_cond)
        cond_u = self._make_unconditional_condition(x_t.shape[0], x_t.device)
        pred_u = model(x_t, cond_u, t_norm, spectral_cond=spectral_cond)
        d = pred_c - pred_u  # [B,M], guidance direction in prediction space

        # map to per-dimension relevance: large |d_j| => relevant connection
        raw = d.abs()
        # normalize per-sample to [0,1]
        raw = raw / (raw.amax(dim=1, keepdim=True) + 1e-8)

        # optional temperature sharpening/softening
        temp = float(self.cfg.prior_mask_temperature)
        if abs(temp - 1.0) > 1e-6:
            raw = raw.pow(1.0 / max(temp, 1e-6))

        return raw.clamp(0.0, 1.0)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z_mean: Optional[np.ndarray] = None,
        z_scale: Optional[np.ndarray] = None,
        subject_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        N, n_features = X.shape

        self.adapter_ = CorrCholeskyAdapter(n_features, z_mean, z_scale)
        n_nodes = self.adapter_.n_nodes

        classes = np.unique(y)
        self.classes_ = classes
        self.n_classes_ = len(classes)
        self.class_map_ = {int(c): i for i, c in enumerate(classes)}
        y_idx = np.array([self.class_map_[int(c)] for c in y], dtype=np.int64)

        if self._no_subtype_mode_enabled():
            cond_idx = self._fit_class_only_tokens(X, y_idx)
        elif self._random_within_class_mode_enabled():
            cond_idx = self._fit_random_within_class_tokens(X, y_idx)
        elif self._global_hard_mode_enabled():
            cond_idx = self._fit_global_subtype_tokens(X, y_idx)
        else:
            # Original current_subtype behavior is intentionally preserved for rollback.
            cond_idx = self._fit_subtype_tokens(X, y_idx)
        sample_weights_np = self._build_sample_weights(subject_ids, N)

        if self._no_subtype_mode_enabled():
            print("[GDT] Class-only conditioning (no subtype partition)")
        elif self._global_hard_mode_enabled():
            print(f"[GDT] Global-GMM subtype counts: {dict(self.class_subtype_counts_)}")
        elif self._random_within_class_mode_enabled():
            print(f"[GDT] Randomized within-class subtype counts: {dict(self.class_subtype_counts_)}")
        else:
            print(f"[GDT] GMM subtypes: {dict(self.class_subtype_counts_)}")
        print(f"[GDT] Conditions: {self.n_conditions_} + 1")
        print(f"[GDT] Subtype conditioning mode: {self.cfg.subtype_conditioning_mode}")
        print(
            f"[GDT] Self-Cond Spectral: {self.cfg.use_spectral_cond}, "
            f"mode={self.cfg.spectral_feature_mode}, "
            f"prob={self.cfg.self_cond_prob}, "
            f"loss_weight={self.cfg.spectral_loss_weight}"
        )
        print(f"[GDT] Denoiser backbone: {self.cfg.denoiser_arch}")
        if self.cfg.anchor_loss_weight > 0:
            print(f"[GDT] Anchor loss enabled (loss_weight={self.cfg.anchor_loss_weight})")
        if self._amortized_bridge_mode_enabled():
            print(
                "[GDT] Spectral mode uses amortized structural bridge inference "
                f"(teacher=internal structural bridge, loss_weight={self.cfg.amortized_bridge_loss_weight})"
            )
        elif self._teacher_bridge_mode_enabled():
            print("[GDT] Spectral mode uses non-amortized teacher-bridge conditioning")
        elif self.cfg.use_spectral_cond:
            print("[GDT] Spectral mode uses self-conditioning only")

        self.model_ = _build_denoiser(
            self.cfg,
            n_nodes=n_nodes,
            n_conditions=self.n_conditions_ + 1,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model_.parameters())
        print(f"[GDT] {n_params:,} parameters")
        if self._bridge_like_mode_enabled():
            self._fit_structural_atoms(X, cond_idx)
            atom_summary = {
                int(token): int(params["atoms"].shape[0])
                for token, params in self.structural_atom_params_.items()
            }
            print(f"[GDT] Structural atoms per subtype: {atom_summary}")

        self._init_schedule()
        self._init_ema()

        x_data = torch.from_numpy(X).to(self.device, dtype=torch.float32)
        cond_data = torch.from_numpy(cond_idx).to(self.device, dtype=torch.long)
        y_data = torch.from_numpy(y_idx).to(self.device, dtype=torch.long)
        sample_weights = torch.from_numpy(sample_weights_np).to(self.device, dtype=torch.float32)

        self._y_idx_train = y_idx
        self._x_train = X  # Store for spectral conditioning
        self._train_cond_idx = cond_idx.copy()

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        def lr_lambda(epoch):
            if epoch < self.cfg.warmup_epochs:
                return (epoch + 1) / self.cfg.warmup_epochs
            decay = max(1, self.cfg.epochs - self.cfg.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * (epoch - self.cfg.warmup_epochs) / decay))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        batch_size = min(self.cfg.batch_size, N)
        steps_per_epoch = max(1, N // batch_size)

        history = {"loss": 0.0}

        for epoch in range(self.cfg.epochs):
            self.model_.train()
            epoch_losses = []
            epoch_spectral_losses = []
            epoch_bridge_losses = []
            epoch_anchor_losses = []

            for _ in range(steps_per_epoch):
                if self.cfg.balanced_sampling:
                    idx = self._get_balanced_indices(y_idx, batch_size)
                else:
                    idx = self.rng.choice(N, batch_size, replace=False)

                idx_t = torch.from_numpy(idx).to(self.device, dtype=torch.long)
                x0 = x_data[idx_t]
                yb = y_data[idx_t]
                cb_hard = cond_data[idx_t]
                wb = sample_weights[idx_t]
                bsz = x0.shape[0]

                t_idx = torch.randint(0, self.cfg.diffusion_steps, (bsz,), device=self.device)
                t_norm = t_idx.float() / (self.cfg.diffusion_steps - 1)
                ab = self.alpha_bars_[t_idx].unsqueeze(-1)
                m_anchor = None
                s_anchor = None

                x_t, noise_eff, m_anchor, s_anchor = self._apply_corruption_law(
                    x0=x0,
                    ab=ab,
                    t_idx=t_idx,
                    t_norm=t_norm,
                    yb=yb,
                    cb_hard=cb_hard,
                )
                if self.cfg.v_prediction:
                    target = torch.sqrt(ab) * noise_eff - torch.sqrt(1 - ab) * x0
                else:
                    target = noise_eff

                cb = cb_hard
                cb_model = cb.clone()
                if self.cfg.cfg_prob > 0:
                    drop_mask = torch.rand(bsz, device=self.device) < self.cfg.cfg_prob
                    cb_model[drop_mask] = self.uncond_idx_

                optimizer.zero_grad(set_to_none=True)

                spectral_cond = None
                bridge_pred = None
                bridge_target = None
                # =============================================================
                # SPECTRAL CONDITIONING
                # =============================================================
                if self.cfg.use_spectral_cond:
                    if self._amortized_bridge_mode_enabled():
                        spectral_cond, bridge_pred = self._build_amortized_bridge_conditioning(
                            x_t,
                            cb_model,
                            t_norm,
                        )
                        with torch.no_grad():
                            _, bridge_target = self._build_mixture_bridge_conditioning(
                                x0,
                                cb_model,
                                ab,
                            )
                    elif self._teacher_bridge_mode_enabled():
                        use_self_cond = torch.rand(1).item() < self.cfg.self_cond_prob
                        if use_self_cond:
                            with torch.no_grad():
                                pred_first = self.model_(x_t, cb_model, t_norm, spectral_cond=None)
                                if self.cfg.v_prediction:
                                    x0_estimate = torch.sqrt(ab) * x_t - torch.sqrt(1 - ab) * pred_first
                                else:
                                    x0_estimate = (x_t - torch.sqrt(1 - ab) * pred_first) / torch.sqrt(ab).clamp(min=1e-8)
                                x0_estimate = x0_estimate.clamp(-self.cfg.clip_sample, self.cfg.clip_sample)
                                spectral_cond, _ = self._build_mixture_bridge_conditioning(
                                    x0_estimate,
                                    cb_model,
                                    ab,
                                )
                    else:
                        use_self_cond = torch.rand(1).item() < self.cfg.self_cond_prob
                        if use_self_cond:
                            with torch.no_grad():
                                pred_first = self.model_(x_t, cb_model, t_norm, spectral_cond=None)
                                if self.cfg.v_prediction:
                                    x0_estimate = torch.sqrt(ab) * x_t - torch.sqrt(1 - ab) * pred_first
                                else:
                                    x0_estimate = (x_t - torch.sqrt(1 - ab) * pred_first) / torch.sqrt(ab).clamp(min=1e-8)
                                x0_estimate = x0_estimate.clamp(-self.cfg.clip_sample, self.cfg.clip_sample)
                                spectral_cond = self.model_.spectral_module(x0_estimate)

                pred = self.model_(x_t, cb_model, t_norm, spectral_cond=spectral_cond)

                mse_per_sample = torch.mean((pred - target) ** 2, dim=1)

                if self.cfg.use_min_snr:
                    snr = ab.squeeze(-1) / (1 - ab.squeeze(-1) + 1e-8)
                    snr_weight = torch.clamp(snr, max=self.cfg.min_snr_gamma) / (snr + 1e-8)
                    mse_per_sample = mse_per_sample * snr_weight

                diffusion_loss = torch.sum(mse_per_sample * wb) / (torch.sum(wb) + 1e-8)

                # =============================================================
                # SPECTRAL CONSISTENCY LOSS: Encourage correct spectral structure
                # =============================================================
                x0_pred = None
                if (
                    (self.cfg.use_spectral_cond and self.cfg.spectral_loss_weight > 0)
                    or self._amortized_bridge_mode_enabled()
                    or self.cfg.anchor_loss_weight > 0
                ):
                    if self.cfg.v_prediction:
                        x0_pred = torch.sqrt(ab) * x_t - torch.sqrt(1 - ab) * pred
                    else:
                        x0_pred = (x_t - torch.sqrt(1 - ab) * pred) / torch.sqrt(ab).clamp(min=1e-8)
                    x0_pred = x0_pred.clamp(-self.cfg.clip_sample, self.cfg.clip_sample)

                spectral_loss = torch.tensor(0.0, device=self.device)
                if self.cfg.use_spectral_cond and self.cfg.spectral_loss_weight > 0:
                    spectral_loss = compute_spectral_consistency_loss(
                        x0_pred,
                        x0,
                        self.model_.spectral_module,
                    )
                weighted_spectral_loss = self.cfg.spectral_loss_weight * spectral_loss

                bridge_loss = torch.tensor(0.0, device=self.device)
                if self._amortized_bridge_mode_enabled():
                    bridge_loss = self._compute_amortized_bridge_loss(
                        bridge_pred=bridge_pred,
                        bridge_target=bridge_target,
                        cond=cb_model,
                        alpha_bar=ab,
                        sample_weights=wb,
                    )
                weighted_bridge_loss = self.cfg.amortized_bridge_loss_weight * bridge_loss

                anchor_loss = torch.tensor(0.0, device=self.device)
                if (
                    self.cfg.anchor_loss_weight > 0
                    and m_anchor is not None
                    and s_anchor is not None
                    and x0_pred is not None
                    and self.adapter_ is not None
                ):
                    corr_pred = self.adapter_.x_to_corr(x0_pred)
                    moment_loss = _anchor_moment_loss(x0_pred, x0, m_anchor)
                    spd_loss = _spd_margin_loss(corr_pred, margin=1e-3)
                    anchor_gate = s_anchor.mean()
                    anchor_loss = anchor_gate * (moment_loss + 0.5 * spd_loss)
                weighted_anchor_loss = self.cfg.anchor_loss_weight * anchor_loss

                total_loss = diffusion_loss + weighted_spectral_loss + weighted_bridge_loss + weighted_anchor_loss
                total_loss.backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.cfg.grad_clip)
                optimizer.step()
                self._update_ema()

                epoch_losses.append(total_loss.item())
                if self.cfg.use_spectral_cond:
                    epoch_spectral_losses.append(spectral_loss.item())
                if self._amortized_bridge_mode_enabled():
                    epoch_bridge_losses.append(bridge_loss.item())
                if self.cfg.anchor_loss_weight > 0:
                    epoch_anchor_losses.append(anchor_loss.item())

            scheduler.step()
            history["loss"] = np.mean(epoch_losses)
            if epoch_spectral_losses:
                history["spectral_loss"] = np.mean(epoch_spectral_losses)
            if epoch_bridge_losses:
                history["bridge_loss"] = np.mean(epoch_bridge_losses)
            if epoch_anchor_losses:
                history["anchor_loss"] = np.mean(epoch_anchor_losses)

            if epoch == 0 or (epoch + 1) % self.cfg.print_every == 0 or epoch == self.cfg.epochs - 1:
                lr = scheduler.get_last_lr()[0]
                spec_str = f" | spec {history.get('spectral_loss', 0):.4f}" if self.cfg.use_spectral_cond else ""
                bridge_str = f" | bridge {history.get('bridge_loss', 0):.4f}" if self._amortized_bridge_mode_enabled() else ""
                anchor_str = f" | anchor {history.get('anchor_loss', 0):.4f}" if self.cfg.anchor_loss_weight > 0 else ""
                print(f"[GDT] epoch {epoch:3d} | lr {lr:.2e} | loss {history['loss']:.4f}{spec_str}{bridge_str}{anchor_str}")

        self._fitted = True

        if self.cfg.use_calibration:
            z_prior_uncal = self._sample_uncalibrated(y)
            self._fit_latent_calibration(X.astype(np.float64), z_prior_uncal, y_idx)
            print(f"[GDT] Calibration fitted")

        return {f"gdt_{k}": v for k, v in history.items()}

    def _get_model(self):
        return self.ema_model_ if self.ema_model_ is not None else self.model_

    @torch.no_grad()
    def _sample_ddim_core(
        self,
        y_class_idx: np.ndarray,
        cond_tokens: Optional[np.ndarray],
        ddim_steps: int,
        cfg_scale: float,
        return_trajectory: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """DDIM sampling with iterative spectral self-conditioning."""
        eta = self.cfg.ddim_eta

        y_t = torch.from_numpy(y_class_idx).to(self.device, dtype=torch.long)
        if cond_tokens is None:
            raise RuntimeError("cond_tokens are required for DDIM sampling.")
        c_t = torch.from_numpy(cond_tokens).to(self.device, dtype=torch.long)
        B = len(y_t)

        model = self._get_model()
        model.eval()

        timesteps = np.linspace(self.cfg.diffusion_steps - 1, 0, ddim_steps, dtype=np.int64)
        timesteps = np.unique(timesteps)[::-1]

        x_t = torch.randn(B, self.adapter_.n_features, device=self.device, dtype=torch.float32)
        trajectory = [] if return_trajectory else None
        if return_trajectory:
            trajectory.append(x_t.detach().cpu().numpy().astype(np.float64))

        # Spectral conditioning is either updated from x0 predictions (teacher bridge)
        # or amortized directly from the current noisy state.
        spectral_cond = None
        bridge_features = None

        for i, t in enumerate(timesteps):
            t_idx = torch.full((B,), int(t), device=self.device, dtype=torch.long)
            t_norm = t_idx.float() / (self.cfg.diffusion_steps - 1)
            ab = self.alpha_bars_[t_idx].unsqueeze(-1)

            c_in = c_t
            if self.cfg.use_spectral_cond and hasattr(model, "spectral_module") and getattr(model, "amortized_bridge_mode", False):
                spectral_cond, bridge_features = self._build_amortized_bridge_conditioning(
                    x_t,
                    c_in,
                    t_norm,
                )
            pred = model(x_t, c_in, t_norm, spectral_cond=spectral_cond)

            # CFG
            if abs(cfg_scale - 1.0) > 1e-6:
                c_uncond = self._make_unconditional_condition(B, x_t.device)
                uncond_spectral = spectral_cond
                if hasattr(model, "spectral_module") and getattr(model, "amortized_bridge_mode", False):
                    uncond_spectral = model.spectral_module(None, batch_size=B, device=x_t.device)
                elif self.cfg.use_spectral_cond and hasattr(model, "spectral_module"):
                    uncond_spectral = model.spectral_module(None, batch_size=B, device=x_t.device)
                pred_uncond = model(x_t, c_uncond, t_norm, spectral_cond=uncond_spectral)

                # Use diversity-preserving CFG if enabled
                if self.div_cfg is not None:
                    pred = self.div_cfg.apply(pred, pred_uncond, cfg_scale)
                else:
                    pred = pred_uncond + cfg_scale * (pred - pred_uncond)

            # Get x0 and eps
            if self.cfg.v_prediction:
                x0_pred = torch.sqrt(ab) * x_t - torch.sqrt(1 - ab) * pred
                eps_pred = torch.sqrt(1 - ab) * x_t + torch.sqrt(ab) * pred
            else:
                eps_pred = pred
                x0_pred = (x_t - torch.sqrt(1 - ab) * eps_pred) / torch.sqrt(ab).clamp(min=1e-8)

            x0_pred = x0_pred.clamp(-self.cfg.clip_sample, self.cfg.clip_sample)

            if i == len(timesteps) - 1:
                x_t = x0_pred
                if return_trajectory:
                    trajectory.append(x_t.detach().cpu().numpy().astype(np.float64))
                break

            t_prev = timesteps[i + 1]
            ab_prev = self.alpha_bars_[torch.full((B,), int(t_prev), device=self.device, dtype=torch.long)].unsqueeze(-1)

            if (
                self.cfg.use_spectral_cond
                and hasattr(model, "spectral_module")
            ):
                if getattr(model, "amortized_bridge_mode", False):
                    spectral_cond = None
                    bridge_features = None
                elif getattr(model, "teacher_bridge_mode", False):
                    spectral_cond, bridge_features = self._build_mixture_bridge_conditioning(
                        x0_pred,
                        c_t,
                        ab_prev,
                        prev_bridge_features=bridge_features,
                    )
                else:
                    spectral_cond = model.spectral_module(x0_pred)

            if eta > 0:
                sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev))
                sigma = sigma.clamp(min=0)
                dir_xt = torch.sqrt(torch.clamp(1 - ab_prev - sigma**2, min=0)) * eps_pred
                noise = torch.randn_like(x_t)
                x_t = torch.sqrt(ab_prev) * x0_pred + dir_xt + sigma * noise
            else:
                x_t = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1 - ab_prev) * eps_pred

            if return_trajectory:
                trajectory.append(x_t.detach().cpu().numpy().astype(np.float64))

        final = x_t.detach().cpu().numpy().astype(np.float64)
        if not return_trajectory:
            return final
        return final, np.stack(trajectory, axis=0)

    def _sample_uncalibrated(self, y_cond: np.ndarray, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        ddim_steps = kwargs.get("ddim_steps", self.cfg.ddim_steps)
        cfg_scale = kwargs.get("cfg_scale", self.cfg.cfg_scale)

        y_cond = np.asarray(y_cond, dtype=np.int64)
        y_idx = np.array([self.class_map_[int(c)] for c in y_cond], dtype=np.int64)
        cond_tokens = self._sample_condition_tokens(y_cond)

        return self._sample_ddim_core(y_idx, cond_tokens, ddim_steps, cfg_scale)

    @torch.no_grad()
    def sample(
        self,
        y_cond: np.ndarray,
        ddim_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        apply_calibration: bool = True,
        return_trajectory: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        ddim_steps = ddim_steps or self.cfg.ddim_steps
        cfg_scale = cfg_scale if cfg_scale is not None else self.cfg.cfg_scale

        y_cond = np.asarray(y_cond, dtype=np.int64)
        y_idx = np.array([self.class_map_[int(c)] for c in y_cond], dtype=np.int64)
        cond_tokens = self._sample_condition_tokens(y_cond)

        sampled = self._sample_ddim_core(
            y_idx,
            cond_tokens,
            ddim_steps,
            cfg_scale,
            return_trajectory=return_trajectory,
        )
        if return_trajectory:
            z, z_traj = sampled
        else:
            z = sampled

        if apply_calibration and self.cfg.use_calibration and self._latent_calibrator is not None:
            z = self._apply_latent_calibration(z, y_idx)

        if return_trajectory:
            return z, z_traj
        return z

    def sample_corr(self, y_cond: np.ndarray, **kwargs) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        sampled = self.sample(y_cond, **kwargs)
        if kwargs.get("return_trajectory", False):
            x, x_traj = sampled
            return self.adapter_.x_to_corr_np(x), self.adapter_.x_to_corr_np(x_traj)
        return self.adapter_.x_to_corr_np(sampled)



__all__ = ["GDTConfig", "GDTPrior", "GraphDiffusionTransformer", "VectorDiffusionMLP"]
