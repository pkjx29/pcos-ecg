"""Turn long-form waveforms into a record-level feature table plus a QC report.

The QC report holds counts only (no IDs, no sample values), so it is safe to share.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from pcos_ecg.signal.features import FEATURE_UNITS, extract_features
from pcos_ecg.signal.peaks import detect_r_peaks
from pcos_ecg.signal.preprocess import bandpass_notch, check_quality


def process_record(x: np.ndarray, fs: float, cfg: dict) -> tuple[dict, list[str], dict[str, str]]:
    """Quality-check, filter and extract one recording.

    Returns (feature values, quality-failure reasons, per-feature failure reasons).
    """
    q, p = cfg["quality"], cfg["preprocessing"]
    quality = check_quality(
        x, fs, q["min_duration_s"], q["max_abs_mv"], q["flatline_s"]
    )
    if not quality.ok:
        values = {name: np.nan for name in FEATURE_UNITS}
        values["n_beats"] = 0
        return values, quality.reasons, {}
    y = bandpass_notch(x, fs, tuple(p["bandpass_hz"]), p.get("notch_hz"), p["filter_order"])
    peaks = detect_r_peaks(y, fs)
    result = extract_features(y, peaks, fs, min_beats=q["min_beats"])
    return result.values, [], result.failures


def build_feature_table(
    waves: pd.DataFrame,
    labels: pd.DataFrame,
    cfg: dict,
    covariates: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    quality_reasons: Counter[str] = Counter()
    feature_failures: Counter[str] = Counter()

    for (subject, record), rec in waves.groupby(["subject_id", "record_id"], sort=True):
        rec = rec.sort_values("sample_index")
        x = rec["amplitude"].to_numpy(dtype=float)
        fs = float(rec["sample_rate_hz"].iloc[0])
        values, q_reasons, f_fail = process_record(x, fs, cfg)
        quality_reasons.update(q_reasons)
        feature_failures.update(f"{k}: {v}" for k, v in f_fail.items())
        rows.append(
            {"subject_id": subject, "record_id": record, "quality_ok": not q_reasons, **values}
        )

    table = pd.DataFrame(rows).merge(labels, on="subject_id", how="inner")
    if covariates is not None:
        # Covariates come from a separate source and are never derived from the ECG.
        table = table.merge(covariates, on="subject_id", how="left")

    ok = table[table["quality_ok"]]
    feature_cols = [c for c in FEATURE_UNITS if c != "n_beats"]
    report = {
        "data_mode": cfg["data_mode"],
        "n_records": int(len(table)),
        "n_records_quality_ok": int(table["quality_ok"].sum()),
        "n_subjects": int(table["subject_id"].nunique()),
        "quality_failure_reasons": dict(quality_reasons),
        "feature_failure_reasons": dict(feature_failures),
        "feature_success_rate": {
            c: round(float(ok[c].notna().mean()), 3) if len(ok) else 0.0 for c in feature_cols
        },
        "feature_units": FEATURE_UNITS,
        "not_computed": {
            "p_wave_dispersion": "needs multiple leads; single-lead input cannot support it"
        },
    }
    return table, report
