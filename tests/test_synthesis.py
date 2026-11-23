import numpy as np
import pandas as pd
import pytest

from pcos_ecg.synthesis.compare import compare_synthesizers
from pcos_ecg.synthesis.generators import ClassConditionalGaussian, make_synthesizer, rows_needed


def _table(n=30, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([1] * 12 + [0] * (n - 12))
    X = pd.DataFrame(
        {
            "HRV_SDNN_Estimate": rng.normal(45 - 10 * y, 5),
            "QTc_Bazett": rng.normal(0.41 + 0.02 * y, 0.01),
            "BMI": rng.normal(23 + 3 * y, 2),
            "Cycle_Regularity": (rng.random(n) < 0.2 + 0.5 * y).astype(float),
        }
    )
    return X, y


def test_rows_needed_tops_up_each_class_to_target():
    y = np.array([0] * 18 + [1] * 12)
    assert rows_needed(y, {0: 80, 1: 70}) == {0: 62, 1: 58}
    assert rows_needed(y, {0: 10, 1: 5}) == {0: 0, 1: 0}  # never negative


def test_class_conditional_gaussian_gives_exact_counts_and_valid_values():
    X, y = _table()
    gen = ClassConditionalGaussian(categorical=["Cycle_Regularity"], seed=1).fit(X, y)
    Xs, ys = gen.sample({0: 62, 1: 58})
    assert (ys == 0).sum() == 62 and (ys == 1).sum() == 58
    assert list(Xs.columns) == list(X.columns)
    assert set(Xs["Cycle_Regularity"]) <= {0.0, 1.0}
    assert Xs["BMI"].between(X["BMI"].min(), X["BMI"].max()).all()  # clipped to training range


def test_class_conditional_gaussian_keeps_class_differences():
    X, y = _table(n=60)
    Xs, ys = ClassConditionalGaussian(seed=2).fit(X, y).sample({0: 300, 1: 300})
    assert Xs.loc[ys == 1, "HRV_SDNN_Estimate"].mean() < Xs.loc[ys == 0, "HRV_SDNN_Estimate"].mean()


def test_sampling_is_reproducible():
    X, y = _table()
    a = ClassConditionalGaussian(seed=3).fit(X, y).sample({0: 5, 1: 5})[0]
    b = ClassConditionalGaussian(seed=3).fit(X, y).sample({0: 5, 1: 5})[0]
    pd.testing.assert_frame_equal(a, b)


def test_column_with_no_values_stays_missing_instead_of_breaking_the_fit():
    X, y = _table()
    X["P_Wave_Dispersion"] = np.nan  # e.g. the scipy backend cannot measure it
    Xs, ys = make_synthesizer("class_conditional_gaussian", seed=0).fit(X, y).sample({0: 4, 1: 4})
    assert Xs["P_Wave_Dispersion"].isna().all()
    assert Xs.drop(columns="P_Wave_Dispersion").notna().all().all()


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        make_synthesizer("bayesian_magic")


def test_sdv_gaussian_copula_respects_class_counts():
    pytest.importorskip("sdv")
    X, y = _table()
    Xs, ys = (
        make_synthesizer("gaussian_copula", ["Cycle_Regularity"], seed=0)
        .fit(X, y)
        .sample({0: 7, 1: 5})
    )
    assert (ys == 0).sum() == 7 and (ys == 1).sum() == 5
    assert list(Xs.columns) == list(X.columns)


def test_comparison_ranks_like_the_study_notebook():
    pytest.importorskip("sdmetrics")
    X, y = _table()
    table = compare_synthesizers(
        X,
        y,
        ["class_conditional_gaussian", "gaussian_copula"],
        n_rows=30,
        categorical=["Cycle_Regularity"],
    )
    assert table["Rank"].tolist() == [1, 2]
    assert table["Overall Quality"].between(0, 1).all()
    assert (table["Overall Quality"].diff().dropna() <= 0).all()
