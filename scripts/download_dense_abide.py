#!/usr/bin/env python3
"""Fetch the ABIDE PCP AAL116/CC200 ROI series used by the atlas study."""

from __future__ import annotations

import argparse
import hashlib
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_abide_pcp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gdt.data import _ensure_dataset_exists  # noqa: E402
ATLASES = {"aal116": ("rois_aal", 116), "cc200": ("rois_cc200", 200)}


def subject_id(value: object) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"Invalid ABIDE subject ID: {value!r}")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    raw = str(value).strip()
    if re.fullmatch(r"[+-]?\d+\.0+", raw):
        return str(int(float(raw)))
    digits = re.findall(r"\d+", raw)
    if not digits or len(digits[-1]) < 5:
        raise ValueError(f"Invalid ABIDE subject ID: {value!r}")
    return str(int(digits[-1]))


def load_series(value: object, n_rois: int) -> np.ndarray | None:
    series = np.loadtxt(value) if isinstance(value, (str, bytes, Path)) else np.asarray(value)
    if series.ndim != 2:
        return None
    if series.shape[0] == n_rois and series.shape[1] != n_rois:
        series = series.T
    if series.shape[1] != n_rois or series.shape[0] < 30:
        return None
    if not np.isfinite(series).all() or np.any(np.var(series, axis=0) <= 1e-12):
        return None
    return np.asarray(series, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlases", default="aal116,cc200")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".nilearn_data")
    args = parser.parse_args()
    requested = [item.strip() for item in args.atlases.split(",") if item.strip()]
    if any(item not in ATLASES for item in requested):
        parser.error(f"Supported atlases: {', '.join(ATLASES)}")

    _ensure_dataset_exists("abide", "msdl")
    with (ROOT / "data_atlas_msdl" / "abide_X_y.pkl").open("rb") as handle:
        reference = pickle.load(handle)
    reference_by_id = {subject_id(row.SubjectID): row for row in reference.itertuples(index=False)}

    # Each atlas keeps its own quality-controlled cohort, matching the paper runs.
    for atlas in requested:
        derivative, n_rois = ATLASES[atlas]
        data = fetch_abide_pcp(
            data_dir=str(args.cache_dir),
            pipeline="cpac",
            derivatives=[derivative],
            band_pass_filtering=True,
            global_signal_regression=False,
            quality_checked=True,
            verbose=1,
        )
        phenotype = pd.DataFrame(data.phenotypic)
        values = list(data[derivative])
        if len(phenotype) != len(values):
            raise RuntimeError("ABIDE phenotype and ROI series counts differ")
        rows = []
        seen = set()
        mismatches = []
        for (_, meta), value in zip(phenotype.iterrows(), values):
            sid = subject_id(meta["SUB_ID"])
            if sid not in reference_by_id:
                continue
            if sid in seen:
                continue
            diagnosis = {1: 1, 2: 0}.get(int(meta["DX_GROUP"]))
            series = load_series(value, n_rois)
            if diagnosis is None or series is None:
                continue
            seen.add(sid)
            reference_row = reference_by_id[sid]
            if diagnosis != int(reference_row.Diagnosis != 0):
                mismatches.append(sid)
            rows.append({
                "SubjectID": reference_row.SubjectID,
                "TimeSeries": series,
                "Diagnosis": int(reference_row.Diagnosis != 0),
                "Age": reference_row.Age,
                "Site": str(meta["SITE_ID"]),
            })
        frame = pd.DataFrame(rows).sort_values("SubjectID").reset_index(drop=True)
        if frame.empty or frame.Diagnosis.nunique() != 2:
            raise RuntimeError(f"No usable binary cohort for {atlas}")
        if mismatches:
            raise RuntimeError(f"ABIDE label mismatch for {len(mismatches)} {atlas} subjects: {mismatches[:10]}")
        destination = ROOT / f"data_atlas_{atlas}"
        destination.mkdir(exist_ok=True)
        frame.to_pickle(destination / "abide_X_y.pkl")
        cohort_hash = hashlib.sha256("\n".join(map(str, frame.SubjectID)).encode()).hexdigest()
        print(f"{atlas}: {len(frame)} subjects, {n_rois} ROIs, cohort_sha256={cohort_hash}")


if __name__ == "__main__":
    main()
