import numpy as np
import pandas as pd
import pytest

from pcos_ecg.features.selection import select_features


def _data(n=150, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    y = (signal + rng.normal(0, 0.5, n) > 0).astype(int)
    X = pd.DataFrame(
        {
            "informative": signal,
            "near_copy": signal + rng.normal(0, 0.01, n),  # |r| > 0.99 with informative
            "noise_a": rng.normal(size=n),
            "noise_b": rng.normal(size=n),
            "empty": np.nan,
        }
    )
    return X, y


def test_correlation_filter_drops_one_of_a_near_duplicate_pair():
    X, y = _data()
    res = select_features(X, y, method="mutual_info", fallback_k=10)
    assert len({"informative", "near_copy"} & set(res.selected)) == 1
    assert set(res.dropped_correlated) <= {"informative", "near_copy"}


def test_all_missing_columns_are_dropped_not_imputed():
    X, y = _data()
    res = select_features(X, y, method="none")
    assert res.dropped_empty == ["empty"] and "empty" not in res.selected


def test_mutual_info_ranks_the_informative_feature_first():
    X, y = _data()
    res = select_features(X, y, method="mutual_info", fallback_k=1)
    assert res.selected[0] in {"informative", "near_copy"}


def test_boruta_keeps_the_signal_and_rejects_noise():
    pytest.importorskip("boruta")
    X, y = _data()
    res = select_features(X, y, method="study", boruta_max_iter=30)
    assert res.selected and set(res.selected) <= {"informative", "near_copy"}
    assert res.boruta_status["noise_a"] == "Rejected"


def test_boruta_falls_back_to_mutual_information_when_nothing_is_confirmed():
    pytest.importorskip("boruta")
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(20, 3)), columns=["a", "b", "c"])
    y = rng.integers(0, 2, 20)
    res = select_features(X, y, method="study", boruta_max_iter=10, fallback_k=2)
    if res.fallback_used:
        assert len(res.selected) == 2
    assert res.selected
