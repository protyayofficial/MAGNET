#!/usr/bin/env python3
"""Train and sample the GDT prior on cross-validation splits."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import math
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from gdt.data import estimate_covariances, load_data
from gdt.pipeline import GDTConfig, GDTPrior


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def _filter_splits_with_all_classes(
    splits,
    y: np.ndarray,
    required_classes: np.ndarray | None = None,
    min_count_per_class: int = 1,
):
    required = set(np.unique(y if required_classes is None else required_classes).tolist())
    filtered = []
    for train_idx, val_idx in splits:
        train_y = y[train_idx]
        val_y = y[val_idx]
        if set(np.unique(train_y).tolist()) < required or set(np.unique(val_y).tolist()) < required:
            continue
        if min_count_per_class > 1:
            train_counts = {int(cls): int(np.sum(train_y == cls)) for cls in required}
            val_counts = {int(cls): int(np.sum(val_y == cls)) for cls in required}
            if min(train_counts.values()) < min_count_per_class or min(val_counts.values()) < min_count_per_class:
                continue
        filtered.append((train_idx, val_idx))
    return filtered


def _corr_to_strict_lower_features(corr: np.ndarray) -> np.ndarray:
    """Map correlation matrices to strict-lower unit-Cholesky features."""
    L = np.linalg.cholesky(corr)
    diag = np.diagonal(L, axis1=-2, axis2=-1)
    L_unit = L / diag[..., :, None]
    tril_i, tril_j = np.tril_indices(corr.shape[-1], k=-1)
    return L_unit[..., tril_i, tril_j]


def _standardize(train_features: np.ndarray, val_features: np.ndarray):
    """Standardize features using training-set statistics only."""
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale = np.clip(scale, 1e-6, None)
    x_train = (train_features - mean) / scale
    x_val = (val_features - mean) / scale
    return x_train.astype(np.float32), x_val.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def _parse_generation_budget_specs(
    fractions_arg: str | None,
    counts_arg: str | None,
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    fractions_arg = (fractions_arg or "").strip()
    counts_arg = (counts_arg or "").strip()

    if not fractions_arg and not counts_arg:
        return [{"key": "", "kind": "full", "value": None, "label": "full"}]

    if fractions_arg:
        for raw in fractions_arg.split(","):
            raw = raw.strip()
            if not raw:
                continue
            frac = float(raw)
            if frac <= 0:
                raise ValueError("Generation budget fractions must be > 0.")
            if math.isclose(frac, 1.0):
                continue
            key = f"budget_frac_{int(round(frac * 1000)):04d}"
            specs.append(
                {
                    "key": key,
                    "kind": "fraction",
                    "value": frac,
                    "label": raw,
                }
            )

    if counts_arg:
        for raw in counts_arg.split(","):
            raw = raw.strip()
            if not raw:
                continue
            count = int(raw)
            if count <= 0:
                raise ValueError("Generation budget counts must be > 0.")
            key = f"budget_count_{count:04d}"
            specs.append(
                {
                    "key": key,
                    "kind": "count",
                    "value": count,
                    "label": raw,
                }
            )

    specs.append({"key": "budget_full", "kind": "full", "value": None, "label": "full"})
    return specs


def _sample_generation_labels(
    y: np.ndarray,
    spec: dict[str, object],
    rng: np.random.RandomState,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    if spec["kind"] == "full":
        return y.copy()

    classes, class_counts = np.unique(y, return_counts=True)
    total = int(len(y))

    if spec["kind"] == "fraction":
        target_total = max(1, int(round(float(spec["value"]) * total)))
    elif spec["kind"] == "count":
        target_total = min(total, int(spec["value"]))
    else:
        raise ValueError(f"Unknown budget kind: {spec['kind']}")

    raw_targets = target_total * class_counts / class_counts.sum()
    class_targets = np.floor(raw_targets).astype(int)
    remainder = target_total - int(class_targets.sum())
    if remainder > 0:
        order = np.argsort(-(raw_targets - class_targets))
        for idx in order[:remainder]:
            class_targets[idx] += 1

    selected_parts = []
    for cls, cls_target in zip(classes, class_targets):
        cls_idx = np.flatnonzero(y == cls)
        cls_target = min(int(cls_target), int(len(cls_idx)))
        if cls_target <= 0:
            continue
        chosen = rng.choice(cls_idx, size=cls_target, replace=False)
        selected_parts.append(y[chosen])

    if not selected_parts:
        chosen = rng.choice(np.arange(total), size=1, replace=False)
        return y[chosen]

    sampled = np.concatenate(selected_parts, axis=0)
    rng.shuffle(sampled)
    return sampled.astype(np.int64, copy=False)


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train manifold-aware GDT.")
    parser.add_argument("--datasets", type=str, default="abide,adni,oasis3")
    parser.add_argument("--atlas", type=str, default="msdl")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="binary",
        choices=["binary", "dataset", "dataset_2plus"],
        help="Use benchmark binary labels or dataset-native labels where available.",
    )
    parser.add_argument(
        "--min-class-count-per-split",
        type=int,
        default=1,
        help="Require at least this many examples of each class in both train and validation.",
    )
    parser.add_argument("--generation-budget-fractions", type=str, default="")
    parser.add_argument("--generation-budget-counts", type=str, default="")
    parser.add_argument(
        "--gdt-save-probe-bundle",
        type=int,
        default=0,
        help="Save a fitted bridge-probe bundle for post-hoc timestep diagnostics.",
    )

    # Naming convention used by the evaluator.
    parser.add_argument("--diffeo-name", type=str, default="corrcholesky")

    # GDT configuration.
    parser.add_argument("--gdt-epochs", type=int, default=200)
    parser.add_argument("--gdt-batch-size", type=int, default=64)
    parser.add_argument("--gdt-lr", type=float, default=1e-4)
    parser.add_argument("--gdt-hidden-dim", type=int, default=128)
    parser.add_argument("--gdt-num-layers", type=int, default=4)
    parser.add_argument("--gdt-num-heads", type=int, default=4)
    parser.add_argument(
        "--gdt-denoiser-arch",
        type=str,
        default="graph_transformer",
        choices=["graph_transformer", "vector_mlp"],
        help="Structural backbone for the diffusion denoiser.",
    )
    parser.add_argument("--gdt-ddim-steps", type=int, default=6)
    parser.add_argument("--gdt-ddim-eta", type=float, default=0.05)
    parser.add_argument("--gdt-cfg-prob", type=float, default=0.15)
    parser.add_argument("--gdt-cfg-scale", type=float, default=1.5)
    parser.add_argument("--gdt-max-subtypes", type=int, default=4)
    parser.add_argument("--gdt-min-samples-subtype", type=int, default=80)
    parser.add_argument(
        "--gdt-subtype-conditioning-mode",
        type=str,
        default="hard",
        choices=["none", "hard", "random_within_class", "global_hard"],
    )
    parser.add_argument("--gdt-use-ema", action="store_true", default=False)
    parser.add_argument("--gdt-use-calibration", action="store_true", default=True)
    parser.add_argument("--gdt-print-every", type=int, default=20)
    parser.add_argument("--gdt-use-pre-norm", action="store_true", default=True)
    parser.add_argument("--gdt-use-qk-norm", action="store_true", default=True)
    parser.add_argument("--gdt-use-min-snr", action="store_true", default=True)
    parser.add_argument("--gdt-min-snr-gamma", type=float, default=5.0)
    parser.add_argument(
        "--gdt-use-spectral-cond",
        nargs="?",
        const=1,
        default=1,
        type=int,
        help="Enable spectral/bridge conditioning. Accepts 0/1 and also works as a bare flag.",
    )
    parser.add_argument(
        "--gdt-no-spectral-cond",
        dest="gdt_use_spectral_cond",
        action="store_const",
        const=0,
        help="Explicitly disable spectral/bridge conditioning.",
    )
    parser.add_argument(
        "--gdt-spectral-feature-mode",
        type=str,
        default="amortized_bridge_v1",
        choices=["amortized_bridge_v1", "self_cond_v1", "teacher_bridge_v1"],
    )
    parser.add_argument("--gdt-self-cond-prob", type=float, default=0.5)
    parser.add_argument("--gdt-spectral-loss-weight", type=float, default=0.1)
    parser.add_argument("--gdt-amortized-bridge-loss-weight", type=float, default=0.1)
    parser.add_argument("--gdt-anchor-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--gdt-use-div-cfg",
        nargs="?",
        const=1,
        default=1,
        type=int,
        help="Enable diversity-preserving CFG. Accepts 0/1 and also works as a bare flag.",
    )
    parser.add_argument(
        "--gdt-no-div-cfg",
        dest="gdt_use_div_cfg",
        action="store_const",
        const=0,
        help="Disable diversity-preserving CFG and fall back to plain CFG.",
    )

    # Prior-based forward noising (relevant dims destroyed slowly)
    parser.add_argument("--gdt-use-prior-slow-noising", type=int, default=1)  # 0/1
    parser.add_argument(
        "--gdt-corruption-law",
        type=str,
        default="adaptive",
        choices=["adaptive", "isotropic", "random_protected", "dense_class_agnostic"],
    )
    parser.add_argument("--gdt-prior-kappa", type=float, default=0.35)
    parser.add_argument("--gdt-prior-power", type=float, default=2.0)
    parser.add_argument("--gdt-prior-temp", type=float, default=1.0)
    return parser.parse_args()


def _build_gdt_config(args: argparse.Namespace) -> GDTConfig:
    return GDTConfig(
        hidden_dim=args.gdt_hidden_dim,
        num_layers=args.gdt_num_layers,
        num_heads=args.gdt_num_heads,
        dropout=0.1,
        use_pre_norm=args.gdt_use_pre_norm,
        use_qk_norm=args.gdt_use_qk_norm,
        denoiser_arch=args.gdt_denoiser_arch,
        use_spectral_cond=bool(args.gdt_use_spectral_cond),
        spectral_feature_mode=args.gdt_spectral_feature_mode,
        self_cond_prob=args.gdt_self_cond_prob,
        spectral_loss_weight=args.gdt_spectral_loss_weight,
        amortized_bridge_loss_weight=args.gdt_amortized_bridge_loss_weight,
        anchor_loss_weight=args.gdt_anchor_loss_weight,
        epochs=args.gdt_epochs,
        batch_size=args.gdt_batch_size,
        lr=args.gdt_lr,
        weight_decay=1e-4,
        grad_clip=1.0,
        warmup_epochs=5,
        diffusion_steps=1000,
        ddim_steps=args.gdt_ddim_steps,
        ddim_eta=args.gdt_ddim_eta,
        schedule="cosine",
        v_prediction=True,
        clip_sample=6.0,
        use_min_snr=args.gdt_use_min_snr,
        min_snr_gamma=args.gdt_min_snr_gamma,
        cfg_prob=args.gdt_cfg_prob,
        cfg_scale=args.gdt_cfg_scale,
        use_diversity_preserving_cfg=args.gdt_use_div_cfg,
        max_subtypes=args.gdt_max_subtypes,
        min_samples_per_subtype=args.gdt_min_samples_subtype,
        gmm_covariance_type="diag",
        gmm_reg_covar=1e-4,
        subtype_conditioning_mode=args.gdt_subtype_conditioning_mode,
        balanced_sampling=True,
        use_ema=args.gdt_use_ema,
        ema_decay=0.999,
        use_calibration=args.gdt_use_calibration,
        calibration_clip=6.0,
        print_every=args.gdt_print_every,
        use_prior_slow_noising=bool(args.gdt_use_prior_slow_noising),
        corruption_law=args.gdt_corruption_law,
        prior_kappa=args.gdt_prior_kappa,
        prior_power=args.gdt_prior_power,
        prior_mask_temperature=args.gdt_prior_temp,
    )


def _run_single_split(
    split_id: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cov: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    budget_specs: list[dict[str, object]],
    budget_output_dirs: dict[str, Path],
    device: str,
) -> None:
    def output_path(budget_dir: Path, name: str) -> Path:
        return budget_dir / f"split_{split_id}_{name}.npy"

    probe_bundle_path = budget_output_dirs[str(budget_specs[0]["key"])] / f"split_{split_id}_bridge_probe_bundle.pt"
    generated_outputs_exist = all(
        output_path(budget_output_dirs[str(spec["key"])], "covariances_generated_samples_val").exists()
        for spec in budget_specs
    )
    probe_bundle_ready = (not bool(args.gdt_save_probe_bundle)) or probe_bundle_path.exists()
    if args.skip_existing and generated_outputs_exist and probe_bundle_ready:
        print(f"Skipping split {split_id} (already exists)")
        return

    _set_seed(args.seed + split_id)

    cov_train, cov_val = cov[train_idx], cov[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    groups_train, groups_val = groups[train_idx], groups[val_idx]

    z_train = _corr_to_strict_lower_features(cov_train)
    z_val = _corr_to_strict_lower_features(cov_val)
    x_train, _x_val, z_mean, z_scale = _standardize(z_train, z_val)

    gdt_config = _build_gdt_config(args)
    prior = GDTPrior(device=device, config=gdt_config, random_state=args.seed + split_id)

    train_start = time.time()
    train_history = prior.fit(x_train, y_train, z_mean=z_mean, z_scale=z_scale, subject_ids=groups_train)
    training_time = time.time() - train_start
    if args.gdt_save_probe_bundle:
        torch.save(prior.export_probe_bundle(), probe_bundle_path)

    subtype_artifacts = prior.export_subtype_artifacts()
    subtype_summary = prior.export_subtype_summary()

    for budget_idx, spec in enumerate(budget_specs):
        budget_dir = budget_output_dirs[str(spec["key"])]
        budget_dir.mkdir(parents=True, exist_ok=True)

        budget_rng = np.random.RandomState(args.seed + 1000 * split_id + 31 * budget_idx)
        y_train_gen = _sample_generation_labels(y_train, spec, budget_rng)
        y_val_gen = _sample_generation_labels(y_val, spec, budget_rng)

        sample_start = time.time()
        sampled_train = prior.sample_corr(y_train_gen)[np.newaxis, ...]
        sampled_val = prior.sample_corr(y_val_gen)[np.newaxis, ...]
        sampling_time = time.time() - sample_start

        np.save(output_path(budget_dir, "covariances_train"), cov_train)
        np.save(output_path(budget_dir, "conditionals_train"), y_train)
        np.save(output_path(budget_dir, "groups_train"), groups_train)
        np.save(output_path(budget_dir, "covariances_val"), cov_val)
        np.save(output_path(budget_dir, "conditionals_val"), y_val)
        np.save(output_path(budget_dir, "groups_val"), groups_val)
        np.save(output_path(budget_dir, "covariances_generated_samples_train"), sampled_train)
        np.save(output_path(budget_dir, "conditionals_generated_samples_train"), y_train_gen)
        np.save(output_path(budget_dir, "covariances_generated_samples_val"), sampled_val)
        np.save(output_path(budget_dir, "conditionals_generated_samples_val"), y_val_gen)
        np.save(output_path(budget_dir, "training_time"), np.array([training_time]))
        np.save(output_path(budget_dir, "sampling_time"), np.array([sampling_time]))
        with open(budget_dir / f"split_{split_id}_train_history.json", "w", encoding="utf-8") as handle:
            json.dump(train_history, handle, indent=2, sort_keys=True)

        with open(budget_dir / f"split_{split_id}_generation_budget.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "budget_key": spec["key"],
                    "budget_kind": spec["kind"],
                    "budget_value": spec["value"],
                    "budget_label": spec["label"],
                    "generated_train_count": int(len(y_train_gen)),
                    "generated_val_count": int(len(y_val_gen)),
                },
                handle,
                indent=2,
                sort_keys=True,
            )

        if subtype_artifacts:
            np.savez_compressed(
                budget_dir / f"split_{split_id}_subtype_artifacts.npz",
                **subtype_artifacts,
            )
        with open(budget_dir / f"split_{split_id}_subtype_summary.json", "w", encoding="utf-8") as handle:
            json.dump(subtype_summary, handle, indent=2, sort_keys=True)


def main() -> None:
    args = _build_args()
    if args.debug:
        args.gdt_epochs = min(args.gdt_epochs, 5)
        args.n_splits = min(args.n_splits, 2)
        args.gdt_print_every = 1

    device = _resolve_device(args.device)
    _set_seed(args.seed)
    print(f"Using device={device}")

    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    datasets = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    budget_specs = _parse_generation_budget_specs(
        args.generation_budget_fractions,
        args.generation_budget_counts,
    )

    optimized_dataset_seeds = {"abide": 21, "adni": 16, "oasis3": 13, "inhouse": 42}

    for dataset in datasets:
        print(f"\n{'=' * 72}\nDataset: {dataset}\n{'=' * 72}")
        dataset_seed = optimized_dataset_seeds.get(dataset, hash(dataset) % (2**31))
        dataset_rng = np.random.RandomState(dataset_seed)
        print(f"Using split seed={dataset_seed}")

        ts, y, groups = load_data(dataset, args.atlas, dataset_rng, label_mode=args.label_mode)
        cov = estimate_covariances(ts, n_jobs=args.n_jobs, normalize=True)

        budget_output_dirs = {}
        for spec in budget_specs:
            budget_key = str(spec["key"])
            if budget_key:
                path = results_root / budget_key / f"{dataset}_{args.atlas}" / "group_None" / f"{args.diffeo_name}_GDT"
            else:
                path = results_root / f"{dataset}_{args.atlas}" / "group_None" / f"{args.diffeo_name}_GDT"
            budget_output_dirs[budget_key] = path
        for path in budget_output_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        if dataset.startswith("inhouse"):
            splitter = StratifiedShuffleSplit(
                n_splits=args.n_splits,
                test_size=max(args.test_size, 0.25),
                random_state=dataset_rng,
            )
            splits = list(splitter.split(cov, y))
        else:
            splitter = GroupShuffleSplit(
                n_splits=args.n_splits * 10,
                test_size=args.test_size,
                random_state=dataset_rng,
            )
            splits = _filter_splits_with_all_classes(
                list(splitter.split(cov, y, groups=groups)),
                y,
                required_classes=np.unique(y),
                min_count_per_class=args.min_class_count_per_split,
            )
            splits = splits[: args.n_splits]

        print(f"Running {len(splits)} splits")
        Parallel(n_jobs=args.n_jobs)(
            delayed(_run_single_split)(
                split_id=split_id,
                train_idx=train_idx,
                val_idx=val_idx,
                cov=cov,
                y=y,
                groups=groups,
                args=args,
                output_dir=results_root,
                budget_specs=budget_specs,
                budget_output_dirs=budget_output_dirs,
                device=device,
            )
            for split_id, (train_idx, val_idx) in enumerate(splits)
        )
        print(f"Finished {dataset}")


if __name__ == "__main__":
    main()
