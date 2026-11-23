"""Turn long-form waveforms into a record-level feature table plus a QC report.

Two extraction backends produce the same study-named columns (features/study.py):
- "neurokit2" (default for the study): NeuroKit2 cleaning, R peaks and wavelet delineation.
- "scipy": the package's own band-pass + Pan-Tompkins + median-beat method. It has no
  per-beat P-wave boundaries, so it leaves P_Wave_Dispersion missing.

The QC report holds counts only (no IDs, no sample values), so it is safe to share.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from pcos_ecg.config import ConfigError
from pcos_ecg.features.study import ECG_MEASURED
from pcos_ecg.signal.features import extract_features
from pcos_ecg.signal.peaks import detect_r_peaks
from pcos_ecg.signal.preprocess import bandpass_notch, check_quality

BACKENDS = ("neurokit2", "scipy")

# scipy-extractor name -> (study name, multiplier to the study's unit)
SCIPY_TO_STUDY = {
    "hrv_sdnn_estimate_s": ("HRV_SDNN_Estimate", 1000.0),
    "mean_rr_s": ("Mean_RR_Interval", 1.0),
    "qt_interval_s": ("QT_Interval", 1.0),
    "qtc_bazett_s": ("QTc_Bazett", 1.0),
    "pr_interval_s": ("PR_Interval", 1.0),
    "qrs_duration_s": ("QRS_Duration", 1.0),
    "t_wave_duration_s": ("T_Wave_Duration", 1.0),
}


def _scipy_features(x: np.ndarray, fs: float, cfg: dict) -> tuple[dict, dict[str, str]]:
    q, p = cfg["quality"], cfg["preprocessing"]
    y = bandpass_notch(x, fs, tuple(p["bandpass_hz"]), p.get("notch_hz"), p["filter_order"])
    result = extract_features(y, detect_r_peaks(y, fs), fs, min_beats=q["min_beats"])
    values = {"n_beats": result.values["n_beats"]}
    failures = {}
    for src, (dst, scale) in SCIPY_TO_STUDY.items():
        values[dst] = result.values[src] * scale
        if src in result.failures:
            failures[dst] = result.failures[src]
    values["P_Wave_Dispersion"] = np.nan
    failures["P_Wave_Dispersion"] = "scipy backend has no per-beat P-wave boundaries"
    return values, failures


def process_record(x: np.ndarray, fs: float, cfg: dict) -> tuple[dict, list[str], dict[str, str]]:
    """Quality-check and extract one recording.

    Returns (feature values, quality-failure reasons, per-feature failure reasons).
    """
    q = cfg["quality"]
    quality = check_quality(x, fs, q["min_duration_s"], q["max_abs_mv"], q["flatline_s"])
    if not quality.ok:
        return {"n_beats": 0, **dict.fromkeys(ECG_MEASURED, np.nan)}, quality.reasons, {}

    backend = (cfg.get("signal") or {}).get("backend", "scipy")
    if backend == "scipy":
        values, failures = _scipy_features(x, fs, cfg)
    elif backend == "neurokit2":
        from pcos_ecg.signal.nk_features import extract_features_nk

        result = extract_features_nk(x, fs, min_beats=q["min_beats"])
        values, failures = result.values, result.failures
    else:
        raise ConfigError(f"signal.backend must be one of {BACKENDS}, got {backend!r}")
    return values, [], failures


def build_feature_table(waves: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """One row per recording with study-named ECG features, plus a counts-only QC report."""
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

    table = pd.DataFrame(rows)
    ok = table[table["quality_ok"]]
    report = {
        "data_mode": cfg["data_mode"],
        "signal_backend": (cfg.get("signal") or {}).get("backend", "scipy"),
        "n_records": int(len(table)),
        "n_records_quality_ok": int(table["quality_ok"].sum()),
        "n_subjects": int(table["subject_id"].nunique()),
        "quality_failure_reasons": dict(quality_reasons),
        "feature_failure_reasons": dict(feature_failures),
        "feature_success_rate": {
            c: round(float(ok[c].notna().mean()), 3) if len(ok) else 0.0 for c in ECG_MEASURED
        },
        "feature_units": ECG_MEASURED,
        "note": "P_Wave_Dispersion is a single-lead proxy (beat-to-beat spread of P duration)",
    }
    return table, report
