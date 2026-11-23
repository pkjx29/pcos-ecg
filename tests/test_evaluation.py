"""Leakage, grouping and metric edge cases for the study-method evaluation."""

import numpy as np
import pandas as pd
import pytest

from pcos_ecg.evaluation.cv import NotEstimableError, make_splits, prepare_table, run_experiment
from pcos_ecg.evaluation.metrics import binary_metrics
from pcos_ecg.models.ensemble import WeightedSoftVote
from pcos_ecg.synthesis.balance import make_balancer
from pcos_ecg.synthesis.generators import make_synthesizer

ECG = ["HRV_SDNN_Estimate", "QTc_Bazett"]
PHYS = ["BMI", "Cycle_Regularity"]


def _table(n_subj=30, n_pos=12, seed=0):
    """One row per participant, with the label carried by both feature groups."""
    rng = np.random.default_rng(seed)
    y = np.array([1] * n_pos + [0] * (n_subj - n_pos))
    return pd.DataFrame(
        {
            "subject_id": [f"P{i:02d}" for i in range(n_subj)],
            "label": y,
            "HRV_SDNN_Estimate": rng.normal(45 - 8 * y, 6),
            "QTc_Bazett": rng.normal(0.41 + 0.01 * y, 0.01),
            "BMI": rng.normal(23 + 3 * y, 2),
            "Cycle_Regularity": (rng.random(n_subj) < 0.2 + 0.5 * y).astype(float),
        }
    )


def _cfg(**over):
    cfg = {
        "data_mode": "synthetic_demo",
        "subject_column": "subject_id",
        "label_column": "label",
        "aggregate_to_subject": None,
        "seed": 0,
        "feature_sets": {"ECG Only": ECG, "Physiology Only": PHYS, "Combined": ECG + PHYS},
        "cv": {"strategy": "group_kfold", "max_splits": 3, "min_subjects_per_class_per_fold": 2},
        "synthesis": {"method": "class_conditional_gaussian", "target_counts": {0: 80, 1: 70}},
        "feature_selection": {"method": "mutual_info", "fallback_k": 2},
        "conditions": ["study_synthesis", "real_only"],
        "models": ["random_forest", "baseline"],
        "ensemble": {"enabled": True, "weights": [0.7, 0.3]},
    }
    cfg.update(over)
    return cfg


def test_synthesizer_only_ever_sees_training_participants():
    df = _table()
    X, y, groups = prepare_table(df, _cfg())
    fitted_on: list[set] = []

    class Spy:
        def __init__(self, *args, **kwargs):
            self.inner = make_synthesizer(*args, **kwargs)

        def fit(self, X_tr, y_tr):
            fitted_on.append({tuple(r) for r in X_tr.round(9).to_numpy()})
            self.inner.fit(X_tr, y_tr)
            return self

        def sample(self, counts):
            return self.inner.sample(counts)

    res = run_experiment(X, y, groups, _cfg(), synthesizer_factory=Spy)
    _, folds = make_splits(y, groups, _cfg()["cv"], 0)
    assert len(fitted_on) == len(folds)
    for seen, (tr, te) in zip(fitted_on, folds, strict=True):
        test_rows = {tuple(r) for r in X.iloc[te].round(9).to_numpy()}
        assert seen.isdisjoint(test_rows)
        assert len(seen) == len(tr)
    assert (res["oof"].query("condition == 'study_synthesis'")["n_synthetic_train_rows"] > 0).all()


def test_every_prediction_is_for_a_real_held_out_participant():
    df = _table()
    X, y, groups = prepare_table(df, _cfg())
    oof = run_experiment(X, y, groups, _cfg())["oof"]
    one = oof.query(
        "condition == 'study_synthesis' and model == 'random_forest' and feature_set == 'Combined'"
    )
    assert sorted(one["row"]) == list(range(len(df)))  # each real participant tested exactly once
    assert (one["y_true"].to_numpy() == y[one["row"].to_numpy()]).all()


def test_leave_one_subject_out_tests_each_person_once():
    df = _table(n_subj=12, n_pos=5)
    X, y, groups = prepare_table(df, _cfg())
    strategy, folds = make_splits(y, groups, {"strategy": "loso"}, 0)
    assert strategy == "leave-one-subject-out" and len(folds) == 12
    assert all(len(te) == 1 for _, te in folds)


def test_auto_strategy_uses_loso_for_small_cohorts():
    df = _table()
    X, y, groups = prepare_table(df, _cfg())
    strategy, _ = make_splits(y, groups, {"strategy": "auto", "loso_max_participants": 60}, 0)
    assert strategy == "leave-one-subject-out"


def test_feature_selection_runs_per_fold_and_is_recorded():
    df = _table()
    X, y, groups = prepare_table(df, _cfg())
    freq = run_experiment(X, y, groups, _cfg())["selection_frequency"]
    assert set(freq["feature_set"]) == {"ECG Only", "Physiology Only", "Combined"}
    assert freq["share"].between(0, 1).all()


def test_ensemble_is_evaluated_and_combines_both_groups():
    df = _table()
    X, y, groups = prepare_table(df, _cfg())
    res = run_experiment(X, y, groups, _cfg())
    assert "ensemble" in set(res["summary"]["model"])
    model = WeightedSoftVote(ECG, PHYS, (0.7, 0.3), seed=0).fit(X, y)
    p = model.predict_proba(X)[:, 1]
    p_ecg = model.ecg_model_.predict_proba(X[ECG])[:, 1]
    p_phys = model.physio_model_.predict_proba(X[PHYS])[:, 1]
    np.testing.assert_allclose(p, 0.7 * p_ecg + 0.3 * p_phys)


def test_ensemble_rejects_weights_that_do_not_sum_to_one():
    X, y, _ = prepare_table(_table(), _cfg())
    with pytest.raises(ValueError, match="sum to 1"):
        WeightedSoftVote(ECG, PHYS, (0.5, 0.2)).fit(X, y)


def test_too_few_real_participants_in_a_class_skips_synthesis_with_warning():
    df = _table()
    X, y, groups = prepare_table(df, _cfg())
    cfg = _cfg(synthesis={"target_counts": {0: 80, 1: 70}, "min_real_per_class": 50})
    res = run_experiment(X, y, groups, cfg)
    assert any("no synthesis" in w for w in res["warnings"])


def test_unknown_condition_is_rejected():
    X, y, groups = prepare_table(_table(), _cfg())
    with pytest.raises(ValueError, match="Unknown conditions"):
        run_experiment(X, y, groups, _cfg(conditions=["magic"]))


def test_missing_configured_feature_is_an_error():
    with pytest.raises(ValueError, match="lacks configured features"):
        prepare_table(_table().drop(columns=["BMI"]), _cfg())


def test_smote_baseline_skipped_with_too_few_minority_participants():
    decision = make_balancer(n_minority_train=3, k_neighbors=3, min_minority=4, seed=0)
    assert decision.sampler is None and "skipped" in decision.skipped_reason


def test_tiny_cohort_is_not_estimable():
    y = np.array([1, 0, 0, 0])
    with pytest.raises(NotEstimableError):
        make_splits(y, np.array(list("abcd")), {"strategy": "loso"}, 0)


def test_single_class_fold_gives_nan_auc():
    m = binary_metrics(np.array([0, 0, 0]), np.array([0.1, 0.7, 0.2]))
    assert np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"]) and np.isnan(m["sensitivity"])
    assert m["specificity"] == pytest.approx(2 / 3)


def test_no_positive_predictions_gives_nan_ppv():
    m = binary_metrics(np.array([1, 0, 1, 0]), np.array([0.1, 0.2, 0.3, 0.4]))
    assert np.isnan(m["ppv"]) and m["sensitivity"] == 0 and m["specificity"] == 1


def test_identifiers_never_become_predictors():
    X, _, _ = prepare_table(_table(), _cfg())
    assert {"subject_id", "label"}.isdisjoint(X.columns)
