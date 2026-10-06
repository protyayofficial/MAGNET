#!/usr/bin/env python3
"""Evaluate MAGNET and externally generated Table 1 split arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gdt.distribution_metrics import alpha_precision, beta_recall  # noqa: E402


METHODS = {
    "corrcholesky_GDT": "MAGNET",
    "corrcholesky_DiffeoCFM": "DiffeoCFM",
    "corrcholesky_DiffeoGauss": "DiffeoGauss",
    "strict_lower_triangular_proj_DiffeoCFM": "TriangCFM",
    "strict_lower_triangular_DiffeoCFM": "TriangCFM",
    "corrcholesky_GDSSProj": "GDSS-proj",
}
METRICS = ["alpha_precision", "beta_recall", "alpha_beta_f1", "roc_auc", "f1", "accuracy", "training_time_s", "sampling_time_s"]


def _array(folder: Path, split: int, suffix: str) -> np.ndarray:
    return np.load(folder / f"split_{split}_{suffix}.npy")


def _last(samples: np.ndarray) -> np.ndarray:
    return samples[-1] if samples.ndim == 4 else samples


def _project_spd(samples: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # Same convex-to-identity projection as the comparison evaluator.
    minimum = np.linalg.eigvalsh(samples).min(axis=-1)
    alpha = np.zeros_like(minimum)
    needs_projection = minimum < eps
    alpha[needs_projection] = (eps - minimum[needs_projection]) / (1 - minimum[needs_projection])
    identity = np.eye(samples.shape[-1])
    return (1 - alpha)[:, None, None] * samples + alpha[:, None, None] * identity


def _quality(real: np.ndarray, generated: np.ndarray) -> tuple[float, float]:
    length = min(len(real), len(generated))
    shape = (length, -1)
    real_flat = real[:length].reshape(shape)
    generated_flat = generated[:length].reshape(shape)
    return (
        alpha_precision(real_flat, generated_flat, random_state=42),
        beta_recall(real_flat, generated_flat, random_state=42),
    )


def _cas(train: np.ndarray, train_y: np.ndarray, val: np.ndarray, val_y: np.ndarray) -> dict[str, float]:
    classifier = make_pipeline(
        TangentSpace(metric="riemann"),
        LogisticRegressionCV(cv=5, penalty="l2", solver="liblinear", class_weight="balanced", random_state=42, max_iter=5000),
    )
    classifier.fit(train, train_y)
    prediction = classifier.predict(val)
    probability = classifier.predict_proba(val)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(val_y, probability)),
        "f1": float(f1_score(val_y, prediction)),
        "accuracy": float(accuracy_score(val_y, prediction)),
    }


def _dataset_label(name: str) -> str:
    dataset, atlas = name.rsplit("_", 1)
    dataset = "OASIS-3" if dataset == "oasis3" else dataset.upper()
    return dataset if atlas == "msdl" else f"{dataset} ({atlas.upper()})"


def _evaluate(folder: Path, split: int, method: str, dataset: str) -> dict[str, float | str | int]:
    real_train = _array(folder, split, "covariances_train")
    real_val = _array(folder, split, "covariances_val")
    train_y = _array(folder, split, "conditionals_train")
    val_y = _array(folder, split, "conditionals_val")
    if method == "Real Data":
        alpha, beta = _quality(real_train, real_val)
        classification = _cas(real_train, train_y, real_val, val_y)
        training_time = sampling_time = float("nan")
    else:
        generated_train = _last(_array(folder, split, "covariances_generated_samples_train"))
        generated_val = _last(_array(folder, split, "covariances_generated_samples_val"))
        generated_y = _array(folder, split, "conditionals_generated_samples_train")
        if np.linalg.eigvalsh(generated_train).min() <= 1e-12 or np.linalg.eigvalsh(generated_val).min() <= 1e-12:
            generated_train = _project_spd(generated_train)
            generated_val = _project_spd(generated_val)
        alpha, beta = _quality(real_val, generated_val)
        classification = _cas(generated_train, generated_y, real_val, val_y)
        training_time = float(_array(folder, split, "training_time").item())
        sampling_time = float(_array(folder, split, "sampling_time").item())
    return {
        "dataset": dataset,
        "method": method,
        "split": split,
        "alpha_precision": alpha,
        "beta_recall": beta,
        "alpha_beta_f1": 2 * alpha * beta / (alpha + beta + 1e-12),
        **classification,
        "training_time_s": training_time,
        "sampling_time_s": sampling_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magnet-results-dir", type=Path, default=ROOT / "results" / "table1")
    parser.add_argument("--baseline-results-dir", type=Path, default=None, help="External baseline split arrays, if available")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "table1_metrics")
    args = parser.parse_args()
    rows = []
    real_splits: dict[tuple[str, int], Path] = {}
    seen_methods: set[tuple[str, str, int]] = set()
    roots = [(args.magnet_results_dir, {"corrcholesky_GDT"})]
    if args.baseline_results_dir is not None:
        roots.append((args.baseline_results_dir, set(METHODS) - {"corrcholesky_GDT"}))
    for root, allowed in roots:
        if not root.is_dir():
            parser.error(f"Results directory does not exist: {root}")
        for folder in sorted(root.rglob("*")):
            if not folder.is_dir() or folder.name not in allowed:
                continue
            dataset_folder = folder.parent.parent
            if "_" not in dataset_folder.name:
                continue
            dataset = _dataset_label(dataset_folder.name)
            for file in sorted(folder.glob("split_*_covariances_val.npy")):
                split = int(file.name.split("_")[1])
                method = METHODS[folder.name]
                method_key = (dataset, method, split)
                if method_key in seen_methods:
                    raise ValueError(f"Duplicate {method} output for {dataset} split {split}: {folder}")
                seen_methods.add(method_key)
                key = (dataset, split)
                if key not in real_splits:
                    rows.append(_evaluate(folder, split, "Real Data", dataset))
                    real_splits[key] = folder
                else:
                    reference = real_splits[key]
                    for suffix in ("covariances_train", "covariances_val", "conditionals_train", "conditionals_val"):
                        if not np.array_equal(_array(folder, split, suffix), _array(reference, split, suffix)):
                            raise ValueError(f"Split mismatch for {dataset} split {split}: {folder} vs {reference}")
                rows.append(_evaluate(folder, split, method, dataset))
                print(f"Evaluated {dataset} {method} split {split}", flush=True)
    if not rows:
        parser.error("No recognized split arrays found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details = pd.DataFrame(rows)
    if args.baseline_results_dir is not None:
        missing = {"DiffeoCFM", "DiffeoGauss", "TriangCFM", "GDSS-proj"} - set(details["method"])
        if missing:
            print(f"Warning: baseline rows absent from comparison: {', '.join(sorted(missing))}")
    details.to_csv(args.output_dir / "split_metrics.csv", index=False)
    summary = details.groupby(["dataset", "method"], sort=True)[METRICS].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.reset_index().to_csv(args.output_dir / "comparison.csv", index=False)
    print(f"Wrote {args.output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
