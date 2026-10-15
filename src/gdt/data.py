"""fMRI data loading and covariance estimation for GDT experiments."""

from __future__ import annotations

import pickle
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.covariance import OAS

try:
    import tabulate
except Exception:  # pragma: no cover - optional dependency
    tabulate = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FMRI_DATASETS = {"abide", "adni", "oasis3", "inhouse"}
DOWNLOADABLE_FMRI_DATASETS = {"abide", "adni", "oasis3"}
FMRI_URL = "https://osf.io/h7sw5/download"


def _atlas_dir(atlas: str) -> Path:
    return PROJECT_ROOT / f"data_atlas_{atlas}"


def _dataset_path(dataset: str, atlas: str) -> Path:
    return _atlas_dir(atlas) / f"{dataset}_X_y.pkl"


def estimate_covariances(ts, normalize: bool = True, n_jobs: int = 1) -> np.ndarray:
    """Estimate covariance/correlation matrices from time series."""
    if (type(ts) not in (list, tuple)) and (type(ts) is np.ndarray and ts.ndim == 2):
        ts = [ts]

    def _cov_est(single_ts: np.ndarray) -> np.ndarray:
        return OAS(store_precision=False).fit(single_ts).covariance_

    if n_jobs == 1:
        cov = np.array([_cov_est(item) for item in ts])
    else:
        cov = np.array(Parallel(n_jobs=n_jobs)(delayed(_cov_est)(item) for item in ts))

    if normalize:
        std = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
        cov = cov / (std[:, :, None] * std[:, None, :])

    return cov


def _ensure_dataset_exists(dataset: str, atlas: str) -> None:
    dataset_path = _dataset_path(dataset, atlas)
    if dataset_path.exists():
        return
    if dataset not in DOWNLOADABLE_FMRI_DATASETS:
        raise FileNotFoundError(
            f"Custom dataset not found: {dataset_path}. Run the in-house preprocessing first."
        )

    if atlas != "msdl":
        raise FileNotFoundError(
            f"{dataset}/{atlas} is not in the MSDL archive. "
            "For ABIDE AAL116/CC200 run: python scripts/download_dense_abide.py"
        )

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = PROJECT_ROOT / f"data_atlas_{atlas}.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(FMRI_URL, zip_path)
    subprocess.run(["unzip", "-o", str(zip_path), "-d", str(PROJECT_ROOT)], check=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found after extraction: {dataset_path}")


def _print_dataset_summary(dataset: str, atlas: str, df, dataset_path: Path) -> None:
    rows = [
        ["Dataset", dataset],
        ["Atlas", atlas],
        ["Subjects", df["SubjectID"].nunique()],
        ["Time series", len(df)],
        ["Regions", df["TimeSeries"].iloc[0].shape[1]],
        ["Classes", df["Diagnosis"].nunique() if "Diagnosis" in df else "N/A"],
        ["Path", dataset_path],
    ]
    if tabulate is not None:
        print(tabulate.tabulate(rows))
    else:
        for key, value in rows:
            print(f"{key}: {value}")


def _encode_dataset_labels(df, dataset: str, label_mode: str):
    if label_mode == "binary":
        return (df["Diagnosis"] != 0).astype(int).to_numpy(), np.ones(len(df), dtype=bool)

    if label_mode not in {"dataset", "dataset_2plus"}:
        raise ValueError(f"Unsupported label_mode: {label_mode}")

    if dataset in {"abide", "inhouse"}:
        return (df["Diagnosis"] != 0).astype(int).to_numpy(), np.ones(len(df), dtype=bool)

    if dataset == "adni":
        if "Group" in df.columns:
            order = ["CN", "SMC", "MCI", "AD"]
            mapping = {label: idx for idx, label in enumerate(order)}
            y = df["Group"].map(mapping)
            if y.isna().any():
                unknown = sorted(df.loc[y.isna(), "Group"].dropna().unique().tolist())
                raise ValueError(f"Unmapped ADNI labels in Group column: {unknown}")
            mask = ~y.isna()
            return y.loc[mask].astype(int).to_numpy(), mask.to_numpy()
        order = [0, 1, 2, 3]
        mapping = {label: idx for idx, label in enumerate(order)}
        y = df["Diagnosis"].map(mapping)
        if y.isna().any():
            unknown = sorted(df.loc[y.isna(), "Diagnosis"].dropna().unique().tolist())
            raise ValueError(f"Unmapped ADNI Diagnosis labels: {unknown}")
        mask = ~y.isna()
        return y.loc[mask].astype(int).to_numpy(), mask.to_numpy()

    if dataset == "oasis3":
        if label_mode == "dataset_2plus":
            order = [0.0, 0.5, 1.0, 2.0, 3.0]
            mapping = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3, 3.0: 3}
            y = df["Diagnosis"].map(mapping)
            mask = ~y.isna()
            return y.loc[mask].astype(int).to_numpy(), mask.to_numpy()
        order = [0.0, 0.5, 1.0, 2.0, 3.0]
        mapping = {label: idx for idx, label in enumerate(order)}
        y = df["Diagnosis"].map(mapping)
        mask = ~y.isna()
        return y.loc[mask].astype(int).to_numpy(), mask.to_numpy()

    raise ValueError(f"Dataset label mode is not defined for dataset: {dataset}")


def load_data(
    dataset: str,
    atlas: str,
    rng: np.random.RandomState,
    verbose: bool = True,
    label_mode: str = "binary",
):
    """Load one fMRI dataset and return ``(time_series, labels, groups)``."""
    dataset = dataset.lower()
    if dataset not in FMRI_DATASETS:
        raise ValueError(f"Unsupported dataset for this experiment: {dataset}")

    _ensure_dataset_exists(dataset, atlas)
    dataset_path = _dataset_path(dataset, atlas)

    with open(dataset_path, "rb") as handle:
        df = pickle.load(handle)

    if verbose:
        _print_dataset_summary(dataset, atlas, df, dataset_path)

    df = df.reset_index(drop=True)
    idx = rng.permutation(np.arange(len(df)))
    df = df.iloc[idx].reset_index(drop=True)

    ts = df["TimeSeries"].values
    y, label_mask = _encode_dataset_labels(df, dataset, label_mode)
    if not np.all(label_mask):
        df = df.loc[label_mask].reset_index(drop=True)
        ts = df["TimeSeries"].values
        groups = df["SubjectID"].values
    else:
        groups = df["SubjectID"].values
    return ts, y, groups
