"""Signal quality checks and filtering for single-lead ECG (amplitude in mV).

Filtering uses sosfiltfilt (forward-backward, zero phase). That is an OFFLINE method:
it needs the whole recording and is not causal, so it does not describe what a
real-time wearable would compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps


class FilterConfigError(ValueError):
    """Raised when a requested filter is impossible at the given sampling rate."""


@dataclass
class QualityResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def check_quality(
    x: np.ndarray,
    fs: float,
    min_duration_s: float = 10.0,
    max_abs_mv: float = 10.0,
    flatline_s: float = 2.0,
) -> QualityResult:
    """Reject recordings that are too short, non-finite, out of range, clipped or flat."""
    reasons = []
    if x.size == 0:
        return QualityResult(False, ["empty signal"])
    if x.size / fs < min_duration_s:
        reasons.append(f"shorter than {min_duration_s:g} s")
    if not np.all(np.isfinite(x)):
        reasons.append("contains non-finite samples")
        return QualityResult(False, reasons)
    if np.max(np.abs(x)) > max_abs_mv:
        reasons.append(f"amplitude beyond {max_abs_mv:g} mV (clipping or wrong unit?)")

    # Saturation: many samples sitting exactly at the extreme value.
    for extreme in (x.max(), x.min()):
        if np.mean(x == extreme) > 0.01:
            reasons.append("signal appears saturated (>1% of samples at an extreme)")
            break

    still = np.abs(np.diff(x)) < 1e-6
    if _longest_run(still) / fs >= flatline_s:
        reasons.append(f"flat segment of {flatline_s:g} s or longer")
    return QualityResult(not reasons, reasons)


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def bandpass_notch(
    x: np.ndarray,
    fs: float,
    bandpass_hz: tuple[float, float] = (0.5, 40.0),
    notch_hz: float | None = 50.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass, then an optional mains notch.

    Raises FilterConfigError instead of quietly changing a cut-off that the sampling rate
    cannot support.
    """
    nyq = fs / 2.0
    low, high = bandpass_hz
    if not 0 < low < high:
        raise FilterConfigError(f"Band-pass must satisfy 0 < low < high, got {bandpass_hz}")
    if high >= nyq:
        raise FilterConfigError(
            f"Band-pass upper edge {high} Hz is not below Nyquist ({nyq} Hz) at fs={fs} Hz"
        )
    sos = sps.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    y = sps.sosfiltfilt(sos, x)

    if notch_hz:
        if notch_hz >= nyq:
            raise FilterConfigError(f"Notch {notch_hz} Hz is not below Nyquist ({nyq} Hz)")
        # Above the band-pass edge the notch is redundant but harmless; keep it for
        # wide-band configurations.
        b, a = sps.iirnotch(notch_hz, Q=30.0, fs=fs)
        y = sps.filtfilt(b, a, y)
    return y
