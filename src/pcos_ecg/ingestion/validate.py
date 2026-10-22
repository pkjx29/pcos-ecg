"""Cohort and waveform checks that report counts only, never sample values or IDs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pcos_ecg.ingestion.load import DataValidationError

MIN_FS_HZ, MAX_FS_HZ = 50.0, 10_000.0


def validate_waveforms(df: pd.DataFrame) -> dict:
    """Structural checks on long-form waveforms. Raises on fatal problems."""
    if df.empty:
        raise DataValidationError("Waveform table is empty")

    issues: list[str] = []
    fs_per_record = df.groupby("record_id")["sample_rate_hz"].nunique()
    if (fs_per_record > 1).any():
        raise DataValidationError(
            f"{int((fs_per_record > 1).sum())} record(s) mix several sampling rates"
        )
    fs = pd.to_numeric(df.groupby("record_id")["sample_rate_hz"].first(), errors="coerce")
    bad_fs = (~np.isfinite(fs)) | (fs < MIN_FS_HZ) | (fs > MAX_FS_HZ)
    if bad_fs.any():
        raise DataValidationError(
            f"{int(bad_fs.sum())} record(s) have an implausible sampling rate "
            f"(allowed {MIN_FS_HZ:g}-{MAX_FS_HZ:g} Hz)"
        )

    owners = df.groupby("record_id")["subject_id"].nunique()
    if (owners > 1).any():
        raise DataValidationError(f"{int((owners > 1).sum())} record_id(s) belong to >1 participant")

    non_monotonic = df.groupby("record_id")["sample_index"].apply(
        lambda s: not s.is_monotonic_increasing or s.duplicated().any()
    )
    if non_monotonic.any():
        issues.append(f"{int(non_monotonic.sum())} record(s) have unsorted or repeated sample_index")

    n_nonfinite = int((~np.isfinite(df["amplitude"].to_numpy(dtype=float))).sum())
    if n_nonfinite:
        issues.append(f"{n_nonfinite} non-finite amplitude sample(s); affected records are flagged")

    return {
        "n_records": int(df["record_id"].nunique()),
        "n_subjects": int(df["subject_id"].nunique()),
        "sample_rates_hz": sorted({float(x) for x in fs.unique()}),
        "issues": issues,
    }


def validate_cohort(labels: pd.DataFrame, record_subjects: pd.Series) -> dict:
    """Cross-check labels against the participants that have recordings."""
    with_data = set(record_subjects.astype(str))
    labelled = set(labels["subject_id"])
    unlabelled, no_data = with_data - labelled, labelled - with_data
    issues = []
    if unlabelled:
        issues.append(f"{len(unlabelled)} participant(s) with recordings have no label; excluded")
    if no_data:
        issues.append(f"{len(no_data)} labelled participant(s) have no recording")

    usable = labels[labels["subject_id"].isin(with_data)]
    counts = usable["label"].value_counts().to_dict()
    n_pos, n_neg = int(counts.get(1, 0)), int(counts.get(0, 0))
    if min(n_pos, n_neg) == 0:
        raise DataValidationError("Need at least one participant in each class")
    return {
        "n_participants_usable": int(len(usable)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "issues": issues,
    }
