"""Signal tests on toy ECG-like signals. They check software behaviour, not medical validity."""

import numpy as np
import pytest

from pcos_ecg.demo.synthetic import SubjectParams, synth_ecg
from pcos_ecg.signal.features import extract_features
from pcos_ecg.signal.peaks import clean_rr, detect_r_peaks
from pcos_ecg.signal.preprocess import FilterConfigError, bandpass_notch, check_quality

FS = 250.0


def _ecg(hr=70, qtc=0.41, pr=0.16, seed=0, duration=30):
    rng = np.random.default_rng(seed)
    return synth_ecg(SubjectParams(hr, 0.03, qtc, pr, 1.1), duration, FS, rng)


def test_filter_rejects_band_above_nyquist():
    with pytest.raises(FilterConfigError):
        bandpass_notch(np.zeros(1000), fs=60.0, bandpass_hz=(0.5, 40.0), notch_hz=None)


def test_filter_rejects_notch_at_nyquist():
    with pytest.raises(FilterConfigError):
        bandpass_notch(np.zeros(1000), fs=100.0, bandpass_hz=(0.5, 40.0), notch_hz=50.0)


def test_filter_removes_mains_and_wander():
    t = np.arange(int(20 * FS)) / FS
    x = 0.5 * np.sin(2 * np.pi * 50 * t) + 1.0 * np.sin(2 * np.pi * 0.1 * t)
    y = bandpass_notch(x, FS)
    assert np.std(y[500:-500]) < 0.02


def test_filter_keeps_length_and_is_deterministic():
    x, _ = _ecg()
    a, b = bandpass_notch(x, FS), bandpass_notch(x, FS)
    assert a.shape == x.shape
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "signal,reason",
    [
        (np.array([]), "empty"),
        (
            np.full(int(5 * FS), 0.1) + np.random.default_rng(0).normal(0, 0.01, int(5 * FS)),
            "shorter",
        ),
        (np.where(np.arange(5000) == 10, np.nan, 0.0), "non-finite"),
        (np.zeros(5000), "flat"),
        (np.random.default_rng(0).normal(0, 50, 5000), "beyond"),
    ],
)
def test_quality_flags(signal, reason):
    q = check_quality(signal, FS)
    assert not q.ok
    assert any(reason in r for r in q.reasons)


def test_clean_signal_passes_quality():
    x, _ = _ecg()
    assert check_quality(x, FS).ok


def test_r_peaks_match_ground_truth():
    x, truth = _ecg(hr=85)
    peaks = detect_r_peaks(bandpass_notch(x, FS), FS)
    assert len(peaks) == len(truth)
    assert np.max(np.abs(peaks - truth)) <= 2  # within 8 ms at 250 Hz


def test_clean_rr_units_are_seconds_and_drop_outliers():
    peaks = np.array([0, 250, 500, 750, 800, 1000, 1250])  # 800 is an extra beat
    rr = clean_rr(peaks, FS)
    assert np.allclose(rr, 1.0)


def test_features_track_generator_parameters():
    x, _ = _ecg(hr=75, qtc=0.42, pr=0.16)
    y = bandpass_notch(x, FS)
    f = extract_features(y, detect_r_peaks(y, FS), FS)
    assert f.failures == {}
    assert f.values["mean_hr_bpm"] == pytest.approx(75, abs=3)
    assert f.values["qtc_bazett_s"] == pytest.approx(0.42, abs=0.03)
    assert f.values["pr_interval_s"] == pytest.approx(0.16, abs=0.03)


def test_too_few_beats_gives_nan_with_reason_not_zero():
    x, _ = _ecg(duration=4)
    y = bandpass_notch(x, FS)
    f = extract_features(y, detect_r_peaks(y, FS), FS, min_beats=8)
    assert np.isnan(f.values["mean_hr_bpm"])
    assert "beats" in f.failures["mean_hr_bpm"]


def test_missing_t_wave_is_reported_not_guessed():
    rng = np.random.default_rng(1)
    x, _ = synth_ecg(SubjectParams(70, 0.03, 0.41, 0.16, 1.1), 30, FS, rng)
    # Remove T and P waves by keeping only +-60 ms around each R peak.
    y = bandpass_notch(x, FS)
    peaks = detect_r_peaks(y, FS)
    mask = np.zeros_like(y, dtype=bool)
    for p in peaks:
        mask[max(p - 15, 0) : p + 15] = True
    y = np.where(mask, y, 0.0)
    f = extract_features(y, peaks, FS)
    assert np.isnan(f.values["qt_interval_s"]) and "qt_interval_s" in f.failures
    assert np.isnan(f.values["pr_interval_s"]) and "pr_interval_s" in f.failures
