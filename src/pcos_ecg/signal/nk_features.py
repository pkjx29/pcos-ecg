"""ECG interval extraction with NeuroKit2: the planned path for the real Kardia recordings.

Steps: nk.ecg_clean (high-pass plus mains-frequency removal) -> nk.ecg_peaks (R peaks) ->
nk.ecg_delineate with the discrete wavelet method (P, QRS and T boundaries per beat).
Each beat's intervals are checked against plausible ranges, then the median over beats is
reported. A feature with too few valid beats is NaN with a reason, never a guess.

P-wave dispersion is classically the spread of P-wave duration across the 12 leads. One
Kardia lead cannot measure that, so `P_Wave_Dispersion` here is a single-lead proxy: the
spread (95th minus 5th percentile) of P-wave duration across beats. It is a different
quantity and is labelled as such in the data dictionary.
"""

from __future__ import annotations

import numpy as np

from pcos_ecg.signal.features import FeatureResult
from pcos_ecg.signal.peaks import clean_rr

# Per-beat plausibility limits in seconds; beats outside are dropped for that feature.
BEAT_RANGES_S = {
    "PR_Interval": (0.08, 0.30),
    "QRS_Duration": (0.04, 0.20),
    "QT_Interval": (0.20, 0.60),
    "T_Wave_Duration": (0.05, 0.40),
    "P_Duration": (0.04, 0.20),
}
FEATURES = (
    "HRV_SDNN_Estimate",
    "Mean_RR_Interval",
    "QT_Interval",
    "QTc_Bazett",
    "PR_Interval",
    "QRS_Duration",
    "P_Wave_Dispersion",
    "T_Wave_Duration",
)


def _nk():
    try:
        import neurokit2 as nk
    except ImportError as exc:  # optional dependency
        raise ImportError(
            "The neurokit2 backend needs the 'study' extra: pip install -e '.[study]'"
        ) from exc
    return nk


def _all_nan(reason: str, n_beats: int = 0) -> FeatureResult:
    return FeatureResult(
        {"n_beats": n_beats, **dict.fromkeys(FEATURES, np.nan)}, dict.fromkeys(FEATURES, reason)
    )


def extract_features_nk(x: np.ndarray, fs: float, min_beats: int = 8) -> FeatureResult:
    """Study-named features from one raw single-lead recording (mV)."""
    nk = _nk()
    clean = nk.ecg_clean(x, sampling_rate=fs, method="neurokit")
    try:
        _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit")
    except Exception as exc:  # NeuroKit2 raises plain exceptions on unusable signals
        return _all_nan(f"R-peak detection failed: {type(exc).__name__}")
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    if peaks.size < min_beats:
        return _all_nan(f"fewer than {min_beats} beats detected", int(peaks.size))

    values: dict[str, float] = dict.fromkeys(FEATURES, np.nan)
    values["n_beats"] = int(peaks.size)
    failures: dict[str, str] = {}

    rr = clean_rr(peaks, fs)
    if rr.size < min_beats - 1:
        return _all_nan(f"fewer than {min_beats - 1} valid RR intervals", int(peaks.size))
    mean_rr = float(np.mean(rr))
    values["Mean_RR_Interval"] = mean_rr
    values["HRV_SDNN_Estimate"] = float(np.std(rr, ddof=1)) * 1000.0  # ms, as in the study

    try:
        _, waves = nk.ecg_delineate(clean, peaks, sampling_rate=fs, method="dwt")
    except Exception as exc:
        reason = f"delineation failed: {type(exc).__name__}"
        for name in ("QT_Interval", "QTc_Bazett", "PR_Interval", "QRS_Duration"):
            failures[name] = reason
        failures["P_Wave_Dispersion"] = failures["T_Wave_Duration"] = reason
        return FeatureResult(values, failures)

    def pts(key: str) -> np.ndarray:
        return np.asarray(waves.get(key, []), dtype=float)

    per_beat = {
        "PR_Interval": pts("ECG_R_Onsets") - pts("ECG_P_Onsets"),
        "QRS_Duration": pts("ECG_R_Offsets") - pts("ECG_R_Onsets"),
        "QT_Interval": pts("ECG_T_Offsets") - pts("ECG_R_Onsets"),
        "T_Wave_Duration": pts("ECG_T_Offsets") - pts("ECG_T_Onsets"),
        "P_Duration": pts("ECG_P_Offsets") - pts("ECG_P_Onsets"),
    }
    valid: dict[str, np.ndarray] = {}
    for name, samples in per_beat.items():
        sec = samples / fs
        lo, hi = BEAT_RANGES_S[name]
        valid[name] = sec[np.isfinite(sec) & (sec >= lo) & (sec <= hi)]

    need = max(3, min_beats // 2)
    for name in ("PR_Interval", "QRS_Duration", "QT_Interval", "T_Wave_Duration"):
        if valid[name].size >= need:
            values[name] = float(np.median(valid[name]))
        else:
            failures[name] = f"fewer than {need} beats with a plausible value"

    if np.isfinite(values["QT_Interval"]):
        values["QTc_Bazett"] = values["QT_Interval"] / np.sqrt(mean_rr)
    else:
        failures["QTc_Bazett"] = failures["QT_Interval"]

    p = valid["P_Duration"]
    if p.size >= min_beats:
        values["P_Wave_Dispersion"] = float(np.percentile(p, 95) - np.percentile(p, 5))
    else:
        failures["P_Wave_Dispersion"] = f"fewer than {min_beats} beats with a measurable P wave"
    return FeatureResult(values, failures)
