"""Per-recording ECG features from a single lead.

Rhythm features come from cleaned RR intervals. Morphology features (PR, QRS, QT, QTc,
T-wave duration) come from a median beat template and a simple threshold method:
a wave boundary is where the template returns to within a fraction of the wave's
amplitude of the isoelectric (PR-segment) level. This is cruder than the tangent method
or expert annotation. A feature that cannot be measured, or falls outside a plausible
range, is returned as NaN with a reason; it is never replaced by a default value.

Not computed: P-wave dispersion (needs several leads; a single lead cannot support it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pcos_ecg.signal.peaks import clean_rr

TEMPLATE_PRE_S, TEMPLATE_POST_S = 0.30, 0.60

# Plausibility limits in seconds; outside these the value is flagged for review.
RANGES_S = {
    "pr_interval_s": (0.08, 0.30),
    "qrs_duration_s": (0.04, 0.20),
    "qt_interval_s": (0.20, 0.60),
    "t_wave_duration_s": (0.05, 0.40),
}

FEATURE_UNITS = {
    "n_beats": "count",
    "mean_hr_bpm": "beats/min",
    "mean_rr_s": "s",
    "hrv_sdnn_estimate_s": "s",
    "hrv_rmssd_s": "s",
    "sdnn_meanrr_ratio": "unitless",
    "pr_interval_s": "s",
    "qrs_duration_s": "s",
    "qt_interval_s": "s",
    "qtc_bazett_s": "s",
    "t_wave_duration_s": "s",
    "r_amplitude_mv": "mV",
}


@dataclass
class FeatureResult:
    values: dict[str, float]
    failures: dict[str, str] = field(default_factory=dict)


def _nan_all(reason: str) -> FeatureResult:
    names = [n for n in FEATURE_UNITS if n != "n_beats"]
    return FeatureResult({"n_beats": 0, **{n: np.nan for n in names}}, dict.fromkeys(names, reason))


def extract_features(
    ecg_filtered: np.ndarray, peaks: np.ndarray, fs: float, min_beats: int = 8
) -> FeatureResult:
    if peaks.size < min_beats:
        return _nan_all(f"fewer than {min_beats} beats detected")

    rr = clean_rr(peaks, fs)
    if rr.size < min_beats - 1:
        return _nan_all(f"fewer than {min_beats - 1} valid RR intervals")

    mean_rr = float(np.mean(rr))
    sdnn = float(np.std(rr, ddof=1))
    values: dict[str, float] = {
        "n_beats": int(peaks.size),
        "mean_hr_bpm": 60.0 / mean_rr,
        "mean_rr_s": mean_rr,
        # Short recordings give a short-term SDNN only, hence "estimate".
        "hrv_sdnn_estimate_s": sdnn,
        "hrv_rmssd_s": float(np.sqrt(np.mean(np.diff(rr) ** 2))),
        "sdnn_meanrr_ratio": sdnn / mean_rr,
    }
    failures: dict[str, str] = {}

    morph, morph_fail = _morphology(ecg_filtered, peaks, fs, mean_rr)
    values.update(morph)
    failures.update(morph_fail)

    qt = values.get("qt_interval_s", np.nan)
    values["qtc_bazett_s"] = qt / np.sqrt(mean_rr) if np.isfinite(qt) else np.nan
    if not np.isfinite(values["qtc_bazett_s"]):
        failures["qtc_bazett_s"] = failures.get("qt_interval_s", "QT not measured")
    return FeatureResult(values, failures)


def median_template(x: np.ndarray, peaks: np.ndarray, fs: float) -> np.ndarray | None:
    pre, post = int(TEMPLATE_PRE_S * fs), int(TEMPLATE_POST_S * fs)
    beats = [x[p - pre : p + post] for p in peaks if p - pre >= 0 and p + post <= x.size]
    if len(beats) < 3:
        return None
    return np.median(np.vstack(beats), axis=0)


def _morphology(
    x: np.ndarray, peaks: np.ndarray, fs: float, mean_rr: float
) -> tuple[dict[str, float], dict[str, str]]:
    names = ["pr_interval_s", "qrs_duration_s", "qt_interval_s", "t_wave_duration_s"]
    out = dict.fromkeys(names, np.nan)
    out["r_amplitude_mv"] = np.nan
    fail: dict[str, str] = {}

    tpl = median_template(x, peaks, fs)
    if tpl is None:
        return out, dict.fromkeys([*names, "r_amplitude_mv"], "too few complete beats for template")

    def idx(t_s: float) -> int:  # time relative to R -> template index
        return int(round((TEMPLATE_PRE_S + t_s) * fs))

    r = idx(0.0)
    iso = float(np.median(tpl[idx(-0.08) : idx(-0.05) + 1]))
    dev = tpl - iso
    r_amp = dev[r]
    out["r_amplitude_mv"] = float(r_amp)
    noise = float(np.std(tpl[idx(-0.08) : idx(-0.05) + 1])) + 1e-6

    # QRS onset: walk back from the Q trough (or R) until within 5% of R amplitude.
    q_lo = idx(-0.06)
    q_trough = q_lo + int(np.argmin(tpl[q_lo:r]))
    start = q_trough if dev[q_trough] < 0 else r
    onset = _walk(dev, start, -1, 0.05 * r_amp, stop=idx(-0.12))
    # QRS offset: walk forward from the S trough (or R) the same way.
    s_hi = idx(0.06)
    s_trough = r + int(np.argmin(tpl[r:s_hi]))
    start = s_trough if dev[s_trough] < 0 else r
    offset = _walk(dev, start, +1, 0.05 * r_amp, stop=idx(0.15))
    if onset is None or offset is None:
        return out, dict.fromkeys(names, "QRS boundaries not found")
    out["qrs_duration_s"] = (offset - onset) / fs

    # T wave: largest deviation between QRS offset + 40 ms and 70% of the RR interval.
    t_lo = offset + int(0.04 * fs)
    t_hi = min(idx(min(0.7 * mean_rr, TEMPLATE_POST_S - 0.02)), tpl.size - 1)
    if t_hi - t_lo < int(0.05 * fs):
        fail["qt_interval_s"] = fail["t_wave_duration_s"] = "no room for T wave search"
    else:
        t_peak = t_lo + int(np.argmax(np.abs(dev[t_lo:t_hi])))
        t_amp = abs(dev[t_peak])
        if t_amp < 3 * noise:
            fail["qt_interval_s"] = fail["t_wave_duration_s"] = "T wave not above noise"
        else:
            t_end = _walk(np.abs(dev), t_peak, +1, 0.1 * t_amp, stop=tpl.size - 1)
            t_start = _walk(np.abs(dev), t_peak, -1, 0.1 * t_amp, stop=offset)
            if t_end is None:
                fail["qt_interval_s"] = "T-wave end not found"
            else:
                out["qt_interval_s"] = (t_end - onset) / fs
            if t_start is None or t_end is None:
                fail["t_wave_duration_s"] = "T-wave boundaries not found"
            else:
                out["t_wave_duration_s"] = (t_end - t_start) / fs

    # P wave: largest positive deflection 300 ms .. 20 ms before QRS onset.
    p_lo, p_hi = max(onset - int(0.30 * fs), 0), onset - int(0.02 * fs)
    p_peak = p_lo + int(np.argmax(dev[p_lo:p_hi])) if p_hi > p_lo else None
    if p_peak is None or dev[p_peak] < 3 * noise:
        fail["pr_interval_s"] = "P wave not above noise"
    else:
        p_on = _walk(dev, p_peak, -1, 0.1 * dev[p_peak], stop=0)
        if p_on is None:
            fail["pr_interval_s"] = "P-wave onset not found"
        else:
            out["pr_interval_s"] = (onset - p_on) / fs

    for name, (lo, hi) in RANGES_S.items():
        v = out[name]
        if np.isfinite(v) and not lo <= v <= hi:
            out[name] = np.nan
            fail[name] = f"outside plausible range {lo}-{hi} s; needs review"
    return out, fail


def _walk(sig: np.ndarray, start: int, step: int, thr: float, stop: int) -> int | None:
    """From `start`, step until |sig| < thr. Returns the index, or None if `stop` is hit."""
    i = start
    while (i - stop) * step < 0:
        if abs(sig[i]) < abs(thr):
            return i
        i += step
    return None
