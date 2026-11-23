import numpy as np
import pandas as pd
import pytest

from pcos_ecg.analysis.outputs import mann_whitney, net_benefit


def test_net_benefit_matches_the_decision_curve_formula():
    y = np.array([1, 1, 0, 0, 0])
    score = np.array([0.9, 0.4, 0.6, 0.1, 0.2])
    nb = net_benefit(y, score, np.array([0.5]))[0]
    # at t = 0.5: TP = 1, FP = 1, n = 5 -> 1/5 - 1/5 * (0.5 / 0.5)
    assert nb == pytest.approx(0.0)
    treat_all = net_benefit(y, np.ones(5), np.array([0.2]))[0]
    assert treat_all == pytest.approx(2 / 5 - 3 / 5 * 0.25)


def test_mann_whitney_flags_a_real_difference_and_handles_empty_features():
    rng = np.random.default_rng(0)
    y = np.array([0] * 15 + [1] * 15)
    X = pd.DataFrame(
        {
            "shifted": np.r_[rng.normal(0, 1, 15), rng.normal(3, 1, 15)],
            "same": rng.normal(0, 1, 30),
            "empty": np.nan,
        }
    )
    table = mann_whitney(X, y).set_index("feature")
    assert table.loc["shifted", "p_value"] < 0.001
    assert table.loc["same", "p_value"] > 0.01
    assert np.isnan(table.loc["empty", "p_value"])
