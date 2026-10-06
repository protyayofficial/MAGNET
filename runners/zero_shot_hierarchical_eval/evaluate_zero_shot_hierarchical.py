#!/usr/bin/env python3
"""Hierarchical zero-shot evaluation over binary-trained generators.

Protocol:
- train a real-data binary classifier: control vs disease
- train a real-data disease-only subclass classifier
- pseudo-label generated train/val samples through that hierarchy
- evaluate:
  1. fidelity of generated val against real val using pseudo subclass assignments
  2. downstream transfer by training a student multiclass classifier on pseudo-labeled
     generated train samples and testing on real val true detailed labels

No generator is retrained.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyriemann.tangentspace import TangentSpace
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.linear_model\._logistic")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ.get("DIFFEO_METRICS_DIR", str(ROOT / "external" / "DiffeoCFM")))

from src.gdt.data import _dataset_path  # noqa: E402
from deterministic_distribution_metrics import alpha_precision, beta_recall  # noqa: E402


ALLOWED_RAW_METHODS = {
    "corrcholesky_GDT",
}

CANONICAL_METHOD_NAMES = {
    "corrcholesky_GDT": "MAGNET",
}

DATASET_ORDER = ["ADNI", "OASIS-3"]
METHOD_ORDER = ["Real Data", "MAGNET"]
SPECTRAL_COV_TOP_K = 5
@dataclass(frozen=True)
class SplitRun:
    source: str
    dataset_raw: str
    dataset: str
    group: str
    raw_method: str
    method: str
    split: int
    method_dir: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchical zero-shot evaluation over binary-trained generators.")
    parser.add_argument(
        "--gdt-results-dir",
        type=Path,
        default=ROOT / "results" / "table1" / "msdl" / "gdt",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "zero_shot_magnet",
    )
    parser.add_argument("--datasets", type=str, default="adni,oasis3")
    parser.add_argument("--atlas", type=str, default="msdl")
    parser.add_argument(
        "--oasis-label-mode",
        type=str,
        choices=("2plus", "raw"),
        default="2plus",
    )
    parser.add_argument("--min-train-per-class", type=int, default=5)
    parser.add_argument("--min-val-per-class", type=int, default=5)
    parser.add_argument(
        "--quality-metrics",
        type=str,
        choices=("hist_only", "full"),
        default="hist_only",
        help=(
            "hist_only reports distribution matching via histogram TV/JS similarity and is much faster. "
            "full additionally computes class-conditioned alpha/beta quality."
        ),
    )
    parser.add_argument("--splits", type=str, default="")
    parser.add_argument("--methods", type=str, default="")
    return parser.parse_args()


def _friendly_dataset_name(raw_dataset: str) -> str:
    dataset = str(raw_dataset)
    if dataset.endswith("_msdl"):
        dataset = dataset[: -len("_msdl")]
    if dataset.lower() == "oasis3":
        return "OASIS-3"
    return dataset.upper()


def _iter_split_runs(base_path: Path, datasets: set[str], source: str) -> list[SplitRun]:
    runs: list[SplitRun] = []
    if not base_path.exists():
        return runs
    for dataset_path in sorted(base_path.iterdir()):
        if not dataset_path.is_dir():
            continue
        dataset_key = dataset_path.name
        while dataset_key and "_" in dataset_key and dataset_key not in datasets:
            dataset_key = "_".join(dataset_key.split("_")[:-1])
        if dataset_key not in datasets:
            continue
        for group_path in sorted(dataset_path.iterdir()):
            if not group_path.is_dir():
                continue
            for method_path in sorted(group_path.iterdir()):
                if not method_path.is_dir():
                    continue
                raw_method = method_path.name
                if raw_method not in ALLOWED_RAW_METHODS:
                    continue
                split_ids = set()
                for file in method_path.glob("split_*_covariances_val.npy"):
                    match = re.match(r"split_(\d+)_", file.name)
                    if match:
                        split_ids.add(int(match.group(1)))
                for split in sorted(split_ids):
                    runs.append(
                        SplitRun(
                            source=source,
                            dataset_raw=dataset_path.name,
                            dataset=_friendly_dataset_name(dataset_path.name),
                            group=group_path.name,
                            raw_method=raw_method,
                            method=CANONICAL_METHOD_NAMES[raw_method],
                            split=split,
                            method_dir=method_path,
                        )
                    )
    return runs


def _final_generated_slice(generated: np.ndarray) -> np.ndarray:
    if generated.ndim == 4:
        return generated[-1]
    if generated.ndim == 3:
        return generated
    raise ValueError(f"Expected generated tensor with shape (T,N,d,d) or (N,d,d), got {generated.shape}")


def _load_split_arrays(run: SplitRun) -> dict[str, Any]:
    def path(name: str) -> Path:
        return run.method_dir / f"split_{run.split}_{name}.npy"

    return {
        "cov_train": np.load(path("covariances_train")),
        "y_train": np.load(path("conditionals_train")),
        "groups_train": np.load(path("groups_train"), allow_pickle=True),
        "cov_val": np.load(path("covariances_val")),
        "y_val": np.load(path("conditionals_val")),
        "groups_val": np.load(path("groups_val"), allow_pickle=True),
        "gen_train_final": _final_generated_slice(np.load(path("covariances_generated_samples_train"))),
        "y_gen_train": np.load(path("conditionals_generated_samples_train")),
        "gen_val_final": _final_generated_slice(np.load(path("covariances_generated_samples_val"))),
        "y_gen_val": np.load(path("conditionals_generated_samples_val")),
        "training_time": float(np.load(path("training_time")).item()),
        "sampling_time": float(np.load(path("sampling_time")).item()),
    }


def _load_subject_label_mapping(dataset: str, atlas: str, oasis_label_mode: str) -> tuple[dict[str, int], dict[int, str]]:
    dataset_key = dataset.lower().replace("-", "")
    with open(_dataset_path(dataset_key, atlas), "rb") as handle:
        df = pickle.load(handle)
    df = df.reset_index(drop=True)

    if dataset_key == "adni":
        if "Group" in df.columns:
            mapping = {"CN": 0, "SMC": 1, "MCI": 2, "AD": 3}
            label_series = df["Group"].map(mapping)
            names = {0: "CN", 1: "SMC", 2: "MCI", 3: "AD"}
        else:
            mapping = {0: 0, 1: 1, 2: 2, 3: 3}
            label_series = df["Diagnosis"].map(mapping)
            names = {0: "CN", 1: "SMC", 2: "MCI", 3: "AD"}
    elif dataset_key == "oasis3":
        if oasis_label_mode == "2plus":
            mapping = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3, 3.0: 3}
            names = {0: "CDR 0", 1: "CDR 0.5", 2: "CDR 1", 3: "CDR 2+"}
        else:
            mapping = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3, 3.0: 4}
            names = {0: "CDR 0", 1: "CDR 0.5", 2: "CDR 1", 3: "CDR 2", 4: "CDR 3"}
        label_series = df["Diagnosis"].map(mapping)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    invalid_mask = label_series.isna()
    if dataset_key == "adni":
        raw_values = df["Group"] if "Group" in df.columns else df["Diagnosis"]
    else:
        raw_values = df["Diagnosis"]
    unknown = sorted(pd.Series(raw_values[invalid_mask]).dropna().unique().tolist())
    if unknown:
        raise ValueError(f"Found unmapped labels in {dataset}: {unknown}")

    tmp = pd.DataFrame({"SubjectID": df.loc[~invalid_mask, "SubjectID"].astype(str), "label": label_series.loc[~invalid_mask].astype(int)})
    subject_to_label = tmp.groupby("SubjectID")["label"].first().astype(int).to_dict()
    return subject_to_label, names


def _subject_labels_and_mask_for_groups(groups: np.ndarray, subject_to_label: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    keep = []
    for item in groups.tolist():
        key = str(item)
        if key in subject_to_label:
            keep.append(True)
            labels.append(int(subject_to_label[key]))
        else:
            keep.append(False)
    return np.asarray(labels, dtype=int), np.asarray(keep, dtype=bool)


def _class_count_map(y: np.ndarray, names: dict[int, str]) -> str:
    values, counts = np.unique(np.asarray(y, dtype=int), return_counts=True)
    return "; ".join(f"{names[int(v)]}:{int(c)}" for v, c in zip(values, counts))


def _project_stack_to_spd_correlation(mats: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mats = np.asarray(mats, dtype=float)
    mats = 0.5 * (mats + np.swapaxes(mats, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(mats)
    eigvals = np.clip(eigvals, eps, None)
    mats = (eigvecs * eigvals[:, None, :]) @ np.swapaxes(eigvecs, -1, -2)
    diag = np.clip(np.diagonal(mats, axis1=-2, axis2=-1), eps, None)
    scales = np.sqrt(diag)
    mats = mats / scales[:, :, None]
    mats = mats / scales[:, None, :]
    mats = 0.5 * (mats + np.swapaxes(mats, -1, -2))
    idx = np.arange(mats.shape[-1])
    mats[:, idx, idx] = 1.0
    return mats


def _upper_triangular_features(mats: np.ndarray) -> np.ndarray:
    mats = _project_stack_to_spd_correlation(mats)
    d = mats.shape[-1]
    tri = np.triu_indices(d, k=1)
    return mats[:, tri[0], tri[1]]


def _make_classifier(y_train: np.ndarray):
    y_train = np.asarray(y_train, dtype=int)
    bincount = np.bincount(y_train)
    present = bincount[bincount > 0]
    min_class = int(np.min(present))
    if min_class >= 5 and len(y_train) >= 20:
        clf = LogisticRegressionCV(
            cv=min(5, min_class),
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
            max_iter=5000,
        )
    else:
        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
            max_iter=5000,
        )
    return make_pipeline(TangentSpace(metric="riemann"), clf)


def _fit_classifier(X_train: np.ndarray, y_train: np.ndarray):
    X_train = _project_stack_to_spd_correlation(X_train)
    model = _make_classifier(y_train)
    model.fit(X_train, y_train)
    return model


def _fit_hierarchical_models(cov_train: np.ndarray, y_train_true: np.ndarray):
    y_train_binary_gate = (y_train_true != 0).astype(int)
    disease_label_values = np.array(sorted(v for v in np.unique(y_train_true) if v != 0), dtype=int)
    disease_remap = {old: new for new, old in enumerate(disease_label_values)}
    y_train_disease = np.asarray([disease_remap[int(v)] for v in y_train_true[y_train_true != 0]], dtype=int)
    binary_model = _fit_classifier(cov_train, y_train_binary_gate)
    disease_model = _fit_classifier(cov_train[y_train_true != 0], y_train_disease)
    return binary_model, disease_model, disease_label_values


def _predict_hierarchical(
    X: np.ndarray,
    binary_model,
    disease_model,
    disease_label_values: np.ndarray,
) -> np.ndarray:
    X = _project_stack_to_spd_correlation(X)
    pred_binary = np.asarray(binary_model.predict(X), dtype=int)
    pred_detailed = np.zeros(len(X), dtype=int)
    disease_mask = pred_binary == 1
    if np.any(disease_mask):
        pred_sub = np.asarray(disease_model.predict(X[disease_mask]), dtype=int)
        pred_detailed[disease_mask] = disease_label_values[pred_sub]
    return pred_detailed


def _class_conditioned_quality(
    x_real: np.ndarray,
    y_real: np.ndarray,
    x_fake: np.ndarray,
    y_fake: np.ndarray,
    label_names: dict[int, str],
    dataset: str,
    method: str,
    split: int,
    label_space: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    classes = sorted(set(np.unique(y_real).tolist()) & set(np.unique(y_fake).tolist()))
    alpha_values: list[float] = []
    beta_values: list[float] = []
    f1_values: list[float] = []
    per_class_rows: list[dict[str, Any]] = []
    for cls in classes:
        real_cls = _project_stack_to_spd_correlation(x_real[y_real == cls])
        fake_cls = _project_stack_to_spd_correlation(x_fake[y_fake == cls])
        if min(len(real_cls), len(fake_cls)) < 5:
            continue
        real_flat = real_cls.reshape(len(real_cls), -1)
        fake_flat = fake_cls.reshape(len(fake_cls), -1)
        alpha = float(alpha_precision(real_flat, fake_flat, plot_curve=False, n_jobs=1, random_state=42))
        beta = float(beta_recall(real_flat, fake_flat, plot_curve=False, n_jobs=1, random_state=42))
        f1 = float(2 * alpha * beta / (alpha + beta + 1e-12))
        alpha_values.append(alpha)
        beta_values.append(beta)
        f1_values.append(f1)
        per_class_rows.append(
            {
                "Dataset": dataset,
                "DetailedLabelSpace": label_space,
                "Method": method,
                "Split": split,
                "Class": int(cls),
                "ClassName": label_names[int(cls)],
                "real_count": int(len(real_cls)),
                "generated_count": int(len(fake_cls)),
                "alpha_precision": alpha,
                "beta_recall": beta,
                "alpha_beta_f1": f1,
            }
        )
    if not f1_values:
        raise ValueError("No classes had enough samples for class-conditioned alpha/beta quality.")
    return (
        {
            "alpha_precision": float(np.mean(alpha_values)),
            "beta_recall": float(np.mean(beta_values)),
            "alpha_beta_f1": float(np.mean(f1_values)),
            "worst_case_beta": float(np.min(beta_values)),
        },
        per_class_rows,
    )


def _distribution_similarity(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    true_hist = np.bincount(y_true, minlength=int(labels.max()) + 1)[labels].astype(float)
    pred_hist = np.bincount(y_pred, minlength=int(labels.max()) + 1)[labels].astype(float)
    true_p = true_hist / max(true_hist.sum(), 1.0)
    pred_p = pred_hist / max(pred_hist.sum(), 1.0)
    tv = 0.5 * float(np.abs(true_p - pred_p).sum())
    m = 0.5 * (true_p + pred_p)
    eps = 1e-12
    kl_true = float(np.sum(true_p * np.log((true_p + eps) / (m + eps))))
    kl_pred = float(np.sum(pred_p * np.log((pred_p + eps) / (m + eps))))
    js = 0.5 * (kl_true + kl_pred) / np.log(2.0)
    return {
        "tv_similarity": float(1.0 - tv),
        "js_similarity": float(1.0 - js),
    }


def _class_js_divergence(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    labels = np.asarray(labels, dtype=int)
    true_hist = np.bincount(y_true, minlength=int(labels.max()) + 1)[labels].astype(float)
    pred_hist = np.bincount(y_pred, minlength=int(labels.max()) + 1)[labels].astype(float)
    true_p = true_hist / max(true_hist.sum(), 1.0)
    pred_p = pred_hist / max(pred_hist.sum(), 1.0)
    m = 0.5 * (true_p + pred_p)
    kl_true = float(np.sum(true_p * np.log((true_p + eps) / (m + eps))))
    kl_pred = float(np.sum(pred_p * np.log((pred_p + eps) / (m + eps))))
    return float(0.5 * (kl_true + kl_pred) / np.log(2.0))


def _ordered_class_emd(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    true_hist = np.bincount(y_true, minlength=int(labels.max()) + 1)[labels].astype(float)
    pred_hist = np.bincount(y_pred, minlength=int(labels.max()) + 1)[labels].astype(float)
    true_p = true_hist / max(true_hist.sum(), 1.0)
    pred_p = pred_hist / max(pred_hist.sum(), 1.0)
    cdf_gap = np.abs(np.cumsum(true_p) - np.cumsum(pred_p))
    max_dist = max(len(labels) - 1, 1)
    return float(np.sum(cdf_gap[:-1]) / max_dist)


def _class_hist_kl_divergence(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    labels = np.asarray(labels, dtype=int)
    true_hist = np.bincount(y_true, minlength=int(labels.max()) + 1)[labels].astype(float)
    pred_hist = np.bincount(y_pred, minlength=int(labels.max()) + 1)[labels].astype(float)
    true_p = true_hist / max(true_hist.sum(), 1.0)
    pred_p = pred_hist / max(pred_hist.sum(), 1.0)
    return float(np.sum(true_p * np.log((true_p + eps) / (pred_p + eps))))


def _class_bhattacharyya_distance(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    labels = np.asarray(labels, dtype=int)
    true_hist = np.bincount(y_true, minlength=int(labels.max()) + 1)[labels].astype(float)
    pred_hist = np.bincount(y_pred, minlength=int(labels.max()) + 1)[labels].astype(float)
    true_p = true_hist / max(true_hist.sum(), 1.0)
    pred_p = pred_hist / max(pred_hist.sum(), 1.0)
    coefficient = float(np.sum(np.sqrt(true_p * pred_p)))
    return float(-np.log(max(coefficient, eps)))


def _within_disease_kl_divergence(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> float:
    disease_labels = np.asarray([int(label) for label in labels if int(label) != 0], dtype=int)
    if disease_labels.size == 0:
        return float("nan")
    return _class_hist_kl_divergence(y_true, y_pred, disease_labels)


def _control_leakage_rate(y_binary_expected: np.ndarray, y_pred_detailed: np.ndarray) -> float:
    y_binary_expected = np.asarray(y_binary_expected, dtype=int)
    y_pred_detailed = np.asarray(y_pred_detailed, dtype=int)
    disease_mask = y_binary_expected == 1
    if not np.any(disease_mask):
        return float("nan")
    return float(np.mean(y_pred_detailed[disease_mask] == 0))


def _median_heuristic_gamma(X: np.ndarray, Y: np.ndarray) -> float:
    pooled = np.concatenate([X, Y], axis=0)
    if len(pooled) < 2:
        return 1.0
    distances = pdist(pooled, metric="sqeuclidean")
    distances = distances[np.isfinite(distances)]
    distances = distances[distances > 0]
    if distances.size == 0:
        return 1.0
    median_sq = float(np.median(distances))
    return 1.0 / max(median_sq, 1e-12)


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))


def _unbiased_mmd_rbf(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if len(X) < 2 or len(Y) < 2:
        return float("nan")
    gamma = _median_heuristic_gamma(X, Y)
    k_xx = _rbf_kernel(X, X, gamma)
    k_yy = _rbf_kernel(Y, Y, gamma)
    k_xy = _rbf_kernel(X, Y, gamma)
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    m = len(X)
    n = len(Y)
    term_xx = k_xx.sum() / (m * (m - 1))
    term_yy = k_yy.sum() / (n * (n - 1))
    term_xy = 2.0 * k_xy.mean()
    return float(max(term_xx + term_yy - term_xy, 0.0))


def _class_conditioned_feature_mmd(x_real: np.ndarray, y_real: np.ndarray, x_fake: np.ndarray, y_fake: np.ndarray) -> float:
    classes = sorted(set(np.unique(y_real).tolist()) & set(np.unique(y_fake).tolist()))
    mmd_values: list[float] = []
    for cls in classes:
        real_cls = np.asarray(x_real[y_real == cls], dtype=float)
        fake_cls = np.asarray(x_fake[y_fake == cls], dtype=float)
        if min(len(real_cls), len(fake_cls)) < 5:
            continue
        real_features = _upper_triangular_features(real_cls)
        fake_features = _upper_triangular_features(fake_cls)
        mmd = _unbiased_mmd_rbf(real_features, fake_features)
        if np.isfinite(mmd):
            mmd_values.append(float(mmd))
    if not mmd_values:
        return float("nan")
    return float(np.mean(mmd_values))


def _class_conditioned_feature_w1(x_real: np.ndarray, y_real: np.ndarray, x_fake: np.ndarray, y_fake: np.ndarray) -> float:
    classes = sorted(set(np.unique(y_real).tolist()) & set(np.unique(y_fake).tolist()))
    w1_values: list[float] = []
    for cls in classes:
        real_cls = np.asarray(x_real[y_real == cls], dtype=float)
        fake_cls = np.asarray(x_fake[y_fake == cls], dtype=float)
        if min(len(real_cls), len(fake_cls)) < 5:
            continue
        real_features = _upper_triangular_features(real_cls)
        fake_features = _upper_triangular_features(fake_cls)
        coord_w1 = [
            wasserstein_distance(real_features[:, idx], fake_features[:, idx])
            for idx in range(real_features.shape[1])
        ]
        w1_values.append(float(np.mean(coord_w1)))
    if not w1_values:
        return float("nan")
    return float(np.mean(w1_values))


def _topk_covariance_spectral_errors(
    X: np.ndarray,
    Y: np.ndarray,
    top_k: int = SPECTRAL_COV_TOP_K,
) -> tuple[float, float]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if len(X) < 2 or len(Y) < 2:
        return float("nan"), float("nan")

    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    rank_cap = min(top_k, Xc.shape[0] - 1, Yc.shape[0] - 1, Xc.shape[1], Yc.shape[1])
    if rank_cap < 1:
        return float("nan"), float("nan")

    _, sx, vtx = np.linalg.svd(Xc, full_matrices=False)
    _, sy, vty = np.linalg.svd(Yc, full_matrices=False)

    eigvals_x = (sx[:rank_cap] ** 2) / max(Xc.shape[0] - 1, 1)
    eigvals_y = (sy[:rank_cap] ** 2) / max(Yc.shape[0] - 1, 1)
    eigval_mse = float(np.mean((eigvals_x - eigvals_y) ** 2))

    eigvec_mse_values: list[float] = []
    for vec_x, vec_y in zip(vtx[:rank_cap], vty[:rank_cap]):
        if float(np.dot(vec_x, vec_y)) < 0.0:
            vec_y = -vec_y
        eigvec_mse_values.append(float(np.mean((vec_x - vec_y) ** 2)))
    eigvec_mse = float(np.mean(eigvec_mse_values))
    return eigval_mse, eigvec_mse


def _class_conditioned_covariance_spectral_metrics(
    x_real: np.ndarray,
    y_real: np.ndarray,
    x_fake: np.ndarray,
    y_fake: np.ndarray,
    top_k: int = SPECTRAL_COV_TOP_K,
) -> dict[str, float]:
    classes = sorted(set(np.unique(y_real).tolist()) & set(np.unique(y_fake).tolist()))
    eigval_values: list[float] = []
    eigvec_values: list[float] = []
    for cls in classes:
        real_cls = np.asarray(x_real[y_real == cls], dtype=float)
        fake_cls = np.asarray(x_fake[y_fake == cls], dtype=float)
        if min(len(real_cls), len(fake_cls)) < 5:
            continue
        real_features = _upper_triangular_features(real_cls)
        fake_features = _upper_triangular_features(fake_cls)
        eigval_mse, eigvec_mse = _topk_covariance_spectral_errors(real_features, fake_features, top_k=top_k)
        if np.isfinite(eigval_mse):
            eigval_values.append(float(eigval_mse))
        if np.isfinite(eigvec_mse):
            eigvec_values.append(float(eigvec_mse))
    return {
        "feature_cov_eigval_mse_top5": float(np.mean(eigval_values)) if eigval_values else float("nan"),
        "feature_cov_eigvec_mse_top5": float(np.mean(eigvec_values)) if eigvec_values else float("nan"),
    }


def _class_conditioned_c2st_auc_gap(x_real: np.ndarray, y_real: np.ndarray, x_fake: np.ndarray, y_fake: np.ndarray) -> float:
    classes = sorted(set(np.unique(y_real).tolist()) & set(np.unique(y_fake).tolist()))
    auc_gaps: list[float] = []
    for cls in classes:
        real_cls = _project_stack_to_spd_correlation(x_real[y_real == cls])
        fake_cls = _project_stack_to_spd_correlation(x_fake[y_fake == cls])
        if min(len(real_cls), len(fake_cls)) < 5:
            continue
        X = np.concatenate([real_cls, fake_cls], axis=0)
        y = np.concatenate(
            [
                np.zeros(len(real_cls), dtype=int),
                np.ones(len(fake_cls), dtype=int),
            ],
            axis=0,
        )
        min_class = int(np.min(np.bincount(y)))
        if min_class < 2:
            continue
        model = make_pipeline(
            TangentSpace(metric="riemann"),
            LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
                max_iter=5000,
            ),
        )
        cv = StratifiedKFold(n_splits=min(3, min_class), shuffle=True, random_state=42)
        prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, prob))
        auc_gaps.append(float(max(auc - 0.5, 0.0) * 2.0))
    if not auc_gaps:
        return float("nan")
    return float(np.mean(auc_gaps))


def _matrix_to_h0_lifetimes(mat: np.ndarray) -> np.ndarray:
    corr = _project_stack_to_spd_correlation(np.asarray(mat, dtype=float)[None, ...])[0]
    dist = 0.5 * (1.0 - corr)
    dist = np.clip(dist, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    mst = minimum_spanning_tree(dist)
    lifetimes = np.sort(np.asarray(mst.data, dtype=np.float64))
    return lifetimes


def _stack_h0_lifetimes(mats: np.ndarray) -> list[np.ndarray]:
    return [_matrix_to_h0_lifetimes(mat) for mat in mats]


def _h0_pi_vector(lifetimes: np.ndarray, grid: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    if lifetimes.size == 0:
        return np.zeros_like(grid)
    delta = (grid[:, None] - lifetimes[None, :]) / max(sigma, 1e-8)
    gauss = np.exp(-0.5 * delta * delta)
    weights = lifetimes[None, :]
    return np.sum(weights * gauss, axis=1, dtype=np.float64)


def _topological_h0_distances(x_real: np.ndarray, y_real: np.ndarray, x_fake: np.ndarray, y_fake: np.ndarray) -> dict[str, float]:
    classes = sorted(set(np.unique(y_real).tolist()) & set(np.unique(y_fake).tolist()))
    w1_values: list[float] = []
    pi_values: list[float] = []
    grid = np.linspace(0.0, 1.0, 48, dtype=np.float64)
    for cls in classes:
        real_cls = np.asarray(x_real[y_real == cls], dtype=float)
        fake_cls = np.asarray(x_fake[y_fake == cls], dtype=float)
        if min(len(real_cls), len(fake_cls)) < 5:
            continue
        real_lifetimes = _stack_h0_lifetimes(real_cls)
        fake_lifetimes = _stack_h0_lifetimes(fake_cls)
        real_pool = np.concatenate(real_lifetimes, axis=0)
        fake_pool = np.concatenate(fake_lifetimes, axis=0)
        w1_values.append(float(wasserstein_distance(real_pool, fake_pool)))
        real_pi = np.mean([_h0_pi_vector(lf, grid) for lf in real_lifetimes], axis=0)
        fake_pi = np.mean([_h0_pi_vector(lf, grid) for lf in fake_lifetimes], axis=0)
        pi_values.append(float(np.mean((real_pi - fake_pi) ** 2)))
    if not w1_values:
        return {"h0_lifetime_w1": float("nan"), "h0_pi_mse": float("nan")}
    return {
        "h0_lifetime_w1": float(np.mean(w1_values)),
        "h0_pi_mse": float(np.mean(pi_values)),
    }


def _student_transfer_metrics(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    model = _fit_classifier(x_train, y_train)
    x_eval = _project_stack_to_spd_correlation(x_eval)
    y_pred = np.asarray(model.predict(x_eval), dtype=int)
    return {
        "f1_macro": float(f1_score(y_eval, y_pred, average="macro", labels=labels, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval, y_pred)),
    }


def _evaluate_real_baseline(
    dataset: str,
    split: int,
    label_names: dict[int, str],
    cov_train: np.ndarray,
    y_train: np.ndarray,
    cov_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    student = _student_transfer_metrics(cov_train, y_train, cov_val, y_val, np.arange(len(label_names), dtype=int))
    label_space = " / ".join(label_names[idx] for idx in sorted(label_names))
    row = {
        "Dataset": dataset,
        "DetailedLabelSpace": label_space,
        "Method": "Real Data",
        "Split": split,
        "tv_similarity": 1.0,
        "js_similarity": 1.0,
        **student,
        "train_time_s": np.nan,
        "sampling_time_s": np.nan,
        "train_class_counts": _class_count_map(y_train, label_names),
        "val_class_counts": _class_count_map(y_val, label_names),
        "n_classes": int(len(label_names)),
    }
    row["alpha_precision"] = 1.0
    row["beta_recall"] = 1.0
    row["alpha_beta_f1"] = 1.0
    row["worst_case_beta"] = 1.0
    row["c2st_auc_gap"] = 0.0
    row["ordered_class_emd"] = 0.0
    row["class_kl_div"] = 0.0
    row["class_bhattacharyya"] = 0.0
    row["class_js_div"] = 0.0
    row["within_disease_kl_div"] = 0.0
    row["control_leakage"] = 0.0
    row["feature_mmd_rbf"] = 0.0
    row["feature_w1_utri"] = 0.0
    row["feature_cov_eigval_mse_top5"] = 0.0
    row["feature_cov_eigvec_mse_top5"] = 0.0
    row["h0_lifetime_w1"] = 0.0
    row["h0_pi_mse"] = 0.0
    class_rows = []
    for cls in sorted(label_names):
        class_rows.append(
            {
                "Dataset": dataset,
                "DetailedLabelSpace": label_space,
                "Method": "Real Data",
                "Split": split,
                "Class": int(cls),
                "ClassName": label_names[int(cls)],
                "real_count": int(np.sum(y_val == cls)),
                "generated_count": int(np.sum(y_val == cls)),
                "alpha_precision": 1.0,
                "beta_recall": 1.0,
                "alpha_beta_f1": 1.0,
            }
        )
    return row, class_rows


def _evaluate_method(
    run: SplitRun,
    label_names: dict[int, str],
    cov_train: np.ndarray,
    y_train_true: np.ndarray,
    cov_val: np.ndarray,
    y_val_true: np.ndarray,
    gen_train: np.ndarray,
    gen_val: np.ndarray,
    y_gen_train_binary: np.ndarray,
    y_gen_val_binary: np.ndarray,
    y_train_binary: np.ndarray,
    y_val_binary: np.ndarray,
    binary_model,
    disease_model,
    disease_label_values: np.ndarray,
    training_time_s: float,
    sampling_time_s: float,
    quality_metrics: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not np.array_equal(np.asarray(y_gen_train_binary, dtype=int), np.asarray(y_train_binary, dtype=int)):
        raise ValueError("Generated train labels do not match saved binary train labels.")
    if not np.array_equal(np.asarray(y_gen_val_binary, dtype=int), np.asarray(y_val_binary, dtype=int)):
        raise ValueError("Generated val labels do not match saved binary val labels.")

    pseudo_train = _predict_hierarchical(gen_train, binary_model, disease_model, disease_label_values)
    pseudo_val = _predict_hierarchical(gen_val, binary_model, disease_model, disease_label_values)

    dist = _distribution_similarity(y_val_true, pseudo_val, np.arange(len(label_names), dtype=int))
    student = _student_transfer_metrics(gen_train, pseudo_train, cov_val, y_val_true, np.arange(len(label_names), dtype=int))
    c2st_auc_gap = _class_conditioned_c2st_auc_gap(cov_val, y_val_true, gen_val, pseudo_val)
    ordered_class_emd = _ordered_class_emd(y_val_true, pseudo_val, np.arange(len(label_names), dtype=int))
    class_kl_div = _class_hist_kl_divergence(y_val_true, pseudo_val, np.arange(len(label_names), dtype=int))
    class_bhattacharyya = _class_bhattacharyya_distance(y_val_true, pseudo_val, np.arange(len(label_names), dtype=int))
    class_js_div = _class_js_divergence(y_val_true, pseudo_val, np.arange(len(label_names), dtype=int))
    within_disease_kl_div = _within_disease_kl_divergence(y_val_true, pseudo_val, np.arange(len(label_names), dtype=int))
    control_leakage = _control_leakage_rate(y_gen_val_binary, pseudo_val)
    feature_mmd_rbf = _class_conditioned_feature_mmd(cov_val, y_val_true, gen_val, pseudo_val)
    feature_w1_utri = _class_conditioned_feature_w1(cov_val, y_val_true, gen_val, pseudo_val)
    cov_spectral = _class_conditioned_covariance_spectral_metrics(cov_val, y_val_true, gen_val, pseudo_val)
    topo = _topological_h0_distances(cov_val, y_val_true, gen_val, pseudo_val)
    label_space = " / ".join(label_names[idx] for idx in sorted(label_names))

    if quality_metrics == "full":
        quality, class_rows = _class_conditioned_quality(
            cov_val,
            y_val_true,
            gen_val,
            pseudo_val,
            label_names=label_names,
            dataset=run.dataset,
            method=run.method,
            split=run.split,
            label_space=label_space,
        )
    else:
        quality = {
            "alpha_precision": np.nan,
            "beta_recall": np.nan,
            "alpha_beta_f1": np.nan,
            "worst_case_beta": np.nan,
        }
        class_rows = []

    return {
        "Dataset": run.dataset,
        "DetailedLabelSpace": label_space,
        "Method": run.method,
        "Split": run.split,
        **quality,
        **dist,
        **student,
        "c2st_auc_gap": float(c2st_auc_gap),
        "ordered_class_emd": float(ordered_class_emd),
        "class_kl_div": float(class_kl_div),
        "class_bhattacharyya": float(class_bhattacharyya),
        "class_js_div": float(class_js_div),
        "within_disease_kl_div": float(within_disease_kl_div),
        "control_leakage": float(control_leakage),
        "feature_mmd_rbf": float(feature_mmd_rbf),
        "feature_w1_utri": float(feature_w1_utri),
        **cov_spectral,
        **topo,
        "train_time_s": float(training_time_s),
        "sampling_time_s": float(sampling_time_s),
        "train_class_counts": _class_count_map(y_train_true, label_names),
        "val_class_counts": _class_count_map(y_val_true, label_names),
        "n_classes": int(len(label_names)),
    }, class_rows


def _aggregate(records: pd.DataFrame) -> pd.DataFrame:
    summary_metrics = [
        "tv_similarity",
        "js_similarity",
        "f1_macro",
        "balanced_accuracy",
        "c2st_auc_gap",
        "ordered_class_emd",
        "class_kl_div",
        "class_bhattacharyya",
        "class_js_div",
        "within_disease_kl_div",
        "control_leakage",
        "feature_mmd_rbf",
        "feature_w1_utri",
        "feature_cov_eigval_mse_top5",
        "feature_cov_eigvec_mse_top5",
        "h0_lifetime_w1",
        "h0_pi_mse",
        "train_time_s",
        "sampling_time_s",
    ]
    if records["alpha_beta_f1"].notna().any():
        summary_metrics = ["alpha_precision", "beta_recall", "alpha_beta_f1", "worst_case_beta"] + summary_metrics
    group_cols = ["Dataset", "DetailedLabelSpace", "Method"]
    agg = records.groupby(group_cols, as_index=False)[summary_metrics + ["n_classes"]].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" if stat else col for col, stat in agg.columns.to_flat_index()]
    split_counts = records.groupby(group_cols).size().reset_index(name="valid_splits")
    meta = (
        records.groupby(group_cols, as_index=False)
        .agg(
            train_class_counts=("train_class_counts", "first"),
            val_class_counts=("val_class_counts", "first"),
        )
        .reset_index(drop=True)
    )
    agg = pd.merge(meta, agg, on=group_cols, how="left")
    agg = pd.merge(agg, split_counts, on=group_cols, how="left")
    for col in agg.columns:
        if col.endswith("_std"):
            agg[col] = agg[col].fillna(0.0)
    return agg


def _paper_summary(summary: pd.DataFrame) -> pd.DataFrame:
    has_alpha_beta = "alpha_precision_mean" in summary.columns
    if has_alpha_beta:
        cols = [
            "Dataset",
            "DetailedLabelSpace",
            "Method",
            "valid_splits",
            "alpha_precision_mean",
            "alpha_precision_std",
            "beta_recall_mean",
            "beta_recall_std",
            "alpha_beta_f1_mean",
            "alpha_beta_f1_std",
            "worst_case_beta_mean",
            "worst_case_beta_std",
            "f1_macro_mean",
            "f1_macro_std",
            "c2st_auc_gap_mean",
            "c2st_auc_gap_std",
            "ordered_class_emd_mean",
            "ordered_class_emd_std",
            "class_kl_div_mean",
            "class_kl_div_std",
            "class_bhattacharyya_mean",
            "class_bhattacharyya_std",
            "class_js_div_mean",
            "class_js_div_std",
            "within_disease_kl_div_mean",
            "within_disease_kl_div_std",
            "control_leakage_mean",
            "control_leakage_std",
            "feature_mmd_rbf_mean",
            "feature_mmd_rbf_std",
            "feature_w1_utri_mean",
            "feature_w1_utri_std",
            "feature_cov_eigval_mse_top5_mean",
            "feature_cov_eigval_mse_top5_std",
            "feature_cov_eigvec_mse_top5_mean",
            "feature_cov_eigvec_mse_top5_std",
            "h0_lifetime_w1_mean",
            "h0_lifetime_w1_std",
            "h0_pi_mse_mean",
            "h0_pi_mse_std",
            "sampling_time_s_mean",
            "sampling_time_s_std",
        ]
    else:
        cols = [
            "Dataset",
            "DetailedLabelSpace",
            "Method",
            "valid_splits",
            "tv_similarity_mean",
            "tv_similarity_std",
            "js_similarity_mean",
            "js_similarity_std",
            "f1_macro_mean",
            "f1_macro_std",
            "sampling_time_s_mean",
            "sampling_time_s_std",
        ]
    return summary[[col for col in cols if col in summary.columns]].copy()


def _compact_summary(paper_df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "alpha_precision_mean": "alpha_precision",
        "alpha_precision_std": "alpha_precision_std",
        "beta_recall_mean": "beta_recall",
        "beta_recall_std": "beta_recall_std",
        "alpha_beta_f1_mean": "alpha_beta_f1",
        "alpha_beta_f1_std": "alpha_beta_f1_std",
        "worst_case_beta_mean": "worst_case_beta",
        "worst_case_beta_std": "worst_case_beta_std",
        "tv_similarity_mean": "tv_similarity",
        "tv_similarity_std": "tv_similarity_std",
        "js_similarity_mean": "js_similarity",
        "js_similarity_std": "js_similarity_std",
        "f1_macro_mean": "cas_f1",
        "f1_macro_std": "cas_f1_std",
        "c2st_auc_gap_mean": "c2st_auc_gap",
        "c2st_auc_gap_std": "c2st_auc_gap_std",
        "ordered_class_emd_mean": "ordered_class_emd",
        "ordered_class_emd_std": "ordered_class_emd_std",
        "class_kl_div_mean": "class_kl_div",
        "class_kl_div_std": "class_kl_div_std",
        "class_bhattacharyya_mean": "class_bhattacharyya",
        "class_bhattacharyya_std": "class_bhattacharyya_std",
        "class_js_div_mean": "class_js_div",
        "class_js_div_std": "class_js_div_std",
        "within_disease_kl_div_mean": "within_disease_kl_div",
        "within_disease_kl_div_std": "within_disease_kl_div_std",
        "control_leakage_mean": "control_leakage",
        "control_leakage_std": "control_leakage_std",
        "feature_mmd_rbf_mean": "feature_mmd_rbf",
        "feature_mmd_rbf_std": "feature_mmd_rbf_std",
        "feature_w1_utri_mean": "feature_w1_utri",
        "feature_w1_utri_std": "feature_w1_utri_std",
        "feature_cov_eigval_mse_top5_mean": "feature_cov_eigval_mse_top5",
        "feature_cov_eigval_mse_top5_std": "feature_cov_eigval_mse_top5_std",
        "feature_cov_eigvec_mse_top5_mean": "feature_cov_eigvec_mse_top5",
        "feature_cov_eigvec_mse_top5_std": "feature_cov_eigvec_mse_top5_std",
        "h0_lifetime_w1_mean": "h0_lifetime_w1",
        "h0_lifetime_w1_std": "h0_lifetime_w1_std",
        "h0_pi_mse_mean": "h0_pi_mse",
        "h0_pi_mse_std": "h0_pi_mse_std",
        "sampling_time_s_mean": "sampling_time_s",
        "sampling_time_s_std": "sampling_time_s_std",
    }
    compact = paper_df.rename(columns=rename_map).copy()
    return compact


def main() -> None:
    args = _parse_args()
    datasets = {d.strip().lower() for d in args.datasets.split(",") if d.strip()}
    unsupported = datasets - {"adni", "oasis3"}
    if unsupported:
        raise ValueError(f"Only ADNI/OASIS3 are supported, got {sorted(unsupported)}")

    selected_splits = None
    if args.splits.strip():
        selected_splits = {int(item.strip()) for item in args.splits.split(",") if item.strip()}
    selected_methods = None
    if args.methods.strip():
        selected_methods = {item.strip() for item in args.methods.split(",") if item.strip()}

    runs = _iter_split_runs(args.gdt_results_dir, datasets, "gdt")
    if selected_splits is not None:
        runs = [run for run in runs if run.split in selected_splits]
    if selected_methods is not None:
        runs = [run for run in runs if run.method in selected_methods]
    if not runs:
        raise SystemExit("No saved runs found.")

    label_maps = {
        dataset: _load_subject_label_mapping(dataset, args.atlas, args.oasis_label_mode)
        for dataset in sorted(datasets)
    }

    raw_records: list[dict[str, Any]] = []
    class_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    emitted_real: set[tuple[str, int]] = set()
    classifier_cache: dict[tuple[str, int], tuple[Any, Any, np.ndarray]] = {}

    for run in runs:
        dataset_key = run.dataset.lower().replace("-", "")
        subject_to_label, label_names = label_maps[dataset_key]
        arrays = _load_split_arrays(run)

        y_train_detailed, train_known_mask = _subject_labels_and_mask_for_groups(arrays["groups_train"], subject_to_label)
        y_val_detailed, val_known_mask = _subject_labels_and_mask_for_groups(arrays["groups_val"], subject_to_label)

        cov_train = arrays["cov_train"][train_known_mask]
        cov_val = arrays["cov_val"][val_known_mask]
        gen_train = arrays["gen_train_final"][train_known_mask]
        gen_val = arrays["gen_val_final"][val_known_mask]
        y_train_binary = np.asarray(arrays["y_train"], dtype=int)[train_known_mask]
        y_val_binary = np.asarray(arrays["y_val"], dtype=int)[val_known_mask]
        y_gen_train_binary = np.asarray(arrays["y_gen_train"], dtype=int)[train_known_mask]
        y_gen_val_binary = np.asarray(arrays["y_gen_val"], dtype=int)[val_known_mask]

        train_counts = np.bincount(y_train_detailed, minlength=len(label_names))
        val_counts = np.bincount(y_val_detailed, minlength=len(label_names))
        if np.any(train_counts < args.min_train_per_class) or np.any(val_counts < args.min_val_per_class):
            skipped_records.append(
                {
                    "Dataset": run.dataset,
                    "Method": run.method,
                    "Split": run.split,
                    "Reason": "insufficient_multiclass_support",
                    "train_class_counts": _class_count_map(y_train_detailed, label_names),
                    "val_class_counts": _class_count_map(y_val_detailed, label_names),
                }
            )
            continue

        cache_key = (run.dataset, run.split)
        if cache_key not in classifier_cache:
            classifier_cache[cache_key] = _fit_hierarchical_models(cov_train, y_train_detailed)
        binary_model, disease_model, disease_label_values = classifier_cache[cache_key]

        real_key = (run.dataset, run.split)
        if real_key not in emitted_real:
            try:
                raw_row, real_class_rows = _evaluate_real_baseline(
                    dataset=run.dataset,
                    split=run.split,
                    label_names=label_names,
                    cov_train=cov_train,
                    y_train=y_train_detailed,
                    cov_val=cov_val,
                    y_val=y_val_detailed,
                )
                raw_records.append(raw_row)
                class_records.extend(real_class_rows)
                emitted_real.add(real_key)
            except Exception as exc:
                skipped_records.append(
                    {
                        "Dataset": run.dataset,
                        "Method": "Real Data",
                        "Split": run.split,
                        "Reason": f"real_eval_error:{type(exc).__name__}",
                        "train_class_counts": _class_count_map(y_train_detailed, label_names),
                        "val_class_counts": _class_count_map(y_val_detailed, label_names),
                    }
                )
                continue

        try:
            raw_row, method_class_rows = _evaluate_method(
                    run=run,
                    label_names=label_names,
                    cov_train=cov_train,
                    y_train_true=y_train_detailed,
                    cov_val=cov_val,
                    y_val_true=y_val_detailed,
                    gen_train=gen_train,
                    gen_val=gen_val,
                    y_gen_train_binary=y_gen_train_binary,
                    y_gen_val_binary=y_gen_val_binary,
                    y_train_binary=y_train_binary,
                    y_val_binary=y_val_binary,
                    binary_model=binary_model,
                    disease_model=disease_model,
                    disease_label_values=disease_label_values,
                    training_time_s=float(arrays["training_time"]),
                    sampling_time_s=float(arrays["sampling_time"]),
                    quality_metrics=args.quality_metrics,
            )
            raw_records.append(raw_row)
            class_records.extend(method_class_rows)
        except Exception as exc:
            skipped_records.append(
                {
                    "Dataset": run.dataset,
                    "Method": run.method,
                    "Split": run.split,
                    "Reason": f"method_eval_error:{type(exc).__name__}",
                    "train_class_counts": _class_count_map(y_train_detailed, label_names),
                    "val_class_counts": _class_count_map(y_val_detailed, label_names),
                }
            )

    if not raw_records:
        raise SystemExit("No valid evaluations were produced.")

    output_root = args.output_root
    tables_dir = output_root / "tables"
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.DataFrame(raw_records)
    raw_df["Dataset"] = pd.Categorical(raw_df["Dataset"], DATASET_ORDER, ordered=True)
    raw_df["Method"] = pd.Categorical(raw_df["Method"], METHOD_ORDER, ordered=True)
    raw_df = raw_df.sort_values(["Dataset", "Method", "Split"]).reset_index(drop=True)
    raw_df.to_csv(output_root / "split_metrics.csv", index=False, float_format="%.6f")
    if class_records:
        class_df = pd.DataFrame(class_records)
        class_df["Dataset"] = pd.Categorical(class_df["Dataset"], DATASET_ORDER, ordered=True)
        class_df["Method"] = pd.Categorical(class_df["Method"], METHOD_ORDER, ordered=True)
        class_df = class_df.sort_values(["Dataset", "Method", "Split", "Class"]).reset_index(drop=True)
        class_df.to_csv(output_root / "class_metrics.csv", index=False, float_format="%.6f")

    summary_df = _aggregate(raw_df)
    summary_df["Dataset"] = pd.Categorical(summary_df["Dataset"], DATASET_ORDER, ordered=True)
    summary_df["Method"] = pd.Categorical(summary_df["Method"], METHOD_ORDER, ordered=True)
    summary_df = summary_df.sort_values(["Dataset", "Method"]).reset_index(drop=True)
    summary_df.to_csv(output_root / "comparison_long.csv", index=False, float_format="%.6f")
    paper_df = _paper_summary(summary_df)
    paper_df.to_csv(output_root / "paper_metrics.csv", index=False, float_format="%.6f")
    _compact_summary(paper_df).to_csv(output_root / "comparison.csv", index=False, float_format="%.6f")

    if skipped_records:
        skipped_df = pd.DataFrame(skipped_records)
        skipped_df["Dataset"] = pd.Categorical(skipped_df["Dataset"], DATASET_ORDER, ordered=True)
        skipped_df["Method"] = pd.Categorical(skipped_df["Method"], METHOD_ORDER, ordered=True)
        skipped_df = skipped_df.sort_values(["Dataset", "Method", "Split"]).reset_index(drop=True)
    else:
        skipped_df = pd.DataFrame(columns=["Dataset", "Method", "Split", "Reason", "train_class_counts", "val_class_counts"])
    skipped_df.to_csv(output_root / "skipped_splits.csv", index=False)

    coverage = raw_df.groupby(["Dataset", "Method"], as_index=False).size().rename(columns={"size": "valid_splits"})
    coverage.to_csv(tables_dir / "split_coverage.csv", index=False)

    print(f"[zero-shot-hierarchical] Wrote split metrics to {output_root / 'split_metrics.csv'}")
    print(f"[zero-shot-hierarchical] Wrote aggregated summary to {output_root / 'comparison_long.csv'}")
    print(f"[zero-shot-hierarchical] Wrote compact summary to {output_root / 'comparison.csv'}")
    print(f"[zero-shot-hierarchical] Wrote paper summary to {output_root / 'paper_metrics.csv'}")
    print(f"[zero-shot-hierarchical] Wrote skipped splits to {output_root / 'skipped_splits.csv'}")


if __name__ == "__main__":
    main()
