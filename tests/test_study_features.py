import numpy as np
import pandas as pd
import pytest

from pcos_ecg.features.study import (
    FINAL_ECG,
    FINAL_PHYSIOLOGICAL,
    add_engineered_features,
    encode_binary,
    feature_sets,
    participant_table,
)


def test_engineered_features_match_formulas():
    df = pd.DataFrame(
        {
            "HRV_SDNN_Estimate": [40.0],  # ms
            "Mean_RR_Interval": [0.8],
            "QTc_Bazett": [0.42],
            "P_Wave_Dispersion": [0.03],
        }
    )
    out = add_engineered_features(df).iloc[0]
    assert out["SDNN_MeanRR_Ratio"] == pytest.approx(0.040 / 0.8)
    assert out["QTc_MeanRR_Ratio"] == pytest.approx(0.42 / 0.8)
    assert out["PWaveDisp_x_QTc"] == pytest.approx(0.03 * 0.42)
    assert out["HRVSDNN_x_QTc"] == pytest.approx(40.0 * 0.42)


def test_zero_rr_gives_nan_not_infinity():
    df = pd.DataFrame(
        {
            "HRV_SDNN_Estimate": [40.0],
            "Mean_RR_Interval": [0.0],
            "QTc_Bazett": [0.4],
            "P_Wave_Dispersion": [0.02],
        }
    )
    out = add_engineered_features(df)
    assert np.isnan(out["SDNN_MeanRR_Ratio"].iloc[0])


def test_binary_answers_are_encoded_and_unknowns_rejected():
    df = pd.DataFrame(
        {"Cycle_Regularity": ["Regular", "irregular "], "Family_History": ["Yes", "No"]}
    )
    out = encode_binary(df)
    assert out["Cycle_Regularity"].tolist() == [0, 1]
    assert out["Family_History"].tolist() == [1, 0]
    with pytest.raises(ValueError, match="unexpected"):
        encode_binary(pd.DataFrame({"Family_History": ["maybe"]}))


def test_participant_table_takes_median_of_good_records():
    records = pd.DataFrame(
        {
            "subject_id": ["A", "A", "A", "B"],
            "quality_ok": [True, True, False, True],
            "HRV_SDNN_Estimate": [30.0, 50.0, 999.0, 40.0],
            "Mean_RR_Interval": [0.8, 0.8, 0.1, 0.9],
            "QTc_Bazett": [0.40, 0.42, 0.9, 0.41],
            "P_Wave_Dispersion": [0.02, 0.04, 0.5, 0.03],
        }
    )
    labels = pd.DataFrame({"subject_id": ["A", "B"], "label": [1, 0]})
    cov = pd.DataFrame(
        {
            "subject_id": ["A", "B"],
            "BMI": [27.0, 21.0],
            "Cycle_Regularity": ["Irregular", "Regular"],
        }
    )
    t = participant_table(records, labels, cov).set_index("subject_id")
    assert t.loc["A", "HRV_SDNN_Estimate"] == 40.0  # the failed record is ignored
    assert t.loc["A", "Cycle_Regularity"] == 1
    assert np.isnan(t.loc["A", "PR_Interval"])  # never measured, so missing rather than 0
    assert len(t) == 2


def test_default_feature_sets_are_the_studys_final_ten():
    sets = feature_sets()
    assert sets["Combined"] == FINAL_ECG + FINAL_PHYSIOLOGICAL
    assert len(sets["Combined"]) == 10
    assert feature_sets({"feature_sets": {"Only BMI": ["BMI"]}}) == {"Only BMI": ["BMI"]}
