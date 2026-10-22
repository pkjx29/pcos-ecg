"""R-peak detection, a simplified Pan-Tompkins (Pan & Tompkins 1985, IEEE TBME 32:230).

Steps: 5-15 Hz band-pass -> derivative -> square -> 150 ms moving-window integration ->
peak picking with a 250 ms refractory period -> refine each peak to the local maximum
of the filtered ECG. Assumes an upright R wave (true for lead II in most people).
This is a research detector, not a clinically validated one.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

MIN_RR_S, MAX_RR_S = 0.3, 2.0  # 200 bpm .. 30 bpm


def detect_r_peaks(ecg_filtered: np.ndarray, fs: float) -> np.ndarray:
    """Return sample indices of R peaks in an already band-passed ECG (mV)."""
    if ecg_filtered.size < int(fs):
        return np.array([], dtype=int)
    sos = sps.butter(2, [5.0, 15.0], btype="bandpass", fs=fs, output="sos")
    qrs_band = sps.sosfiltfilt(sos, ecg_filtered)
    energy = np.gradient(qrs_band) ** 2
    win = max(1, int(0.150 * fs))
    integrated = np.convolve(energy, np.ones(win) / win, mode="same")

    height = 0.3 * np.percentile(integrated, 98)
    candidates, _ = sps.find_peaks(integrated, height=height, distance=int(0.25 * fs))

    half = int(0.075 * fs)
    peaks = []
    for c in candidates:
        lo, hi = max(c - half, 0), min(c + half + 1, ecg_filtered.size)
        peaks.append(lo + int(np.argmax(ecg_filtered[lo:hi])))
    return np.unique(np.array(peaks, dtype=int))


def clean_rr(peaks: np.ndarray, fs: float) -> np.ndarray:
    """RR intervals in seconds, dropping physiologically implausible ones and those more
    than 20% away from the median (likely missed or extra beats)."""
    rr = np.diff(peaks) / fs
    rr = rr[(rr >= MIN_RR_S) & (rr <= MAX_RR_S)]
    if rr.size == 0:
        return rr
    med = np.median(rr)
    return rr[np.abs(rr - med) <= 0.2 * med]
