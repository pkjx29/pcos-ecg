"""Both extraction backends on synthetic ECGs. These check software behaviour, not accuracy
on real recordings."""

import numpy as np
import pytest

from pcos_ecg.config import ConfigError
from pcos_ecg.features.build import process_record
from pcos_ecg.features.study import ECG_MEASURED

FS = 250.0
CFG = {
    "quality": {"min_duration_s": 10, "max_abs_mv": 10.0, "flatline_s": 2.0, "min_beats": 8},
    "preprocessing": {"bandpass_hz": [0.5, 40.0], "notch_hz": 50.0, "filter_order": 4},
}


@pytest.fixture(scope="module")
def nk_ecg():
    nk = pytest.importorskip("neurokit2")
    return nk.ecg_simulate(duration=30, sampling_rate=int(FS), heart_rate=70, random_state=1)


def test_neurokit2_backend_gives_plausible_study_features(nk_ecg):
    from pcos_ecg.signal.nk_features import extract_features_nk

    res = extract_features_nk(np.asarray(nk_ecg), FS)
    v = res.values
    assert v["n_beats"] >= 30
    assert 60_000 / 80 / 1000 < v["Mean_RR_Interval"] < 60 / 60  # 60-80 bpm
    assert 0 <= v["HRV_SDNN_Estimate"] < 200  # ms
    assert 0.08 <= v["PR_Interval"] <= 0.30
    assert 0.30 <= v["QTc_Bazett"] <= 0.60
    assert v["P_Wave_Dispersion"] >= 0  # single-lead spread of P duration


def test_neurokit2_backend_reports_too_few_beats_as_missing(nk_ecg):
    from pcos_ecg.signal.nk_features import extract_features_nk

    res = extract_features_nk(np.asarray(nk_ecg)[: int(3 * FS)], FS, min_beats=8)
    assert np.isnan(res.values["QTc_Bazett"])
    assert "beats" in res.failures["QTc_Bazett"]


@pytest.mark.parametrize("backend", ["scipy", "neurokit2"])
def test_backends_return_the_same_study_columns(backend, nk_ecg):
    values, q_reasons, _ = process_record(
        np.asarray(nk_ecg), FS, {**CFG, "signal": {"backend": backend}}
    )
    assert q_reasons == []
    assert set(ECG_MEASURED) <= set(values)


def test_scipy_backend_leaves_p_wave_dispersion_missing_with_reason(nk_ecg):
    _, _, failures = process_record(np.asarray(nk_ecg), FS, {**CFG, "signal": {"backend": "scipy"}})
    assert "P-wave" in failures["P_Wave_Dispersion"]


def test_unknown_backend_is_a_config_error(nk_ecg):
    with pytest.raises(ConfigError):
        process_record(np.asarray(nk_ecg), FS, {**CFG, "signal": {"backend": "matlab"}})
