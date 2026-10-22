"""Leakage, grouping and metric edge cases for the model comparison."""

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from pcos_ecg.evaluation.cv import NotEstimableError, choose_n_splits, prepare_xy, run_experiment
from pcos_ecg.evaluation.metrics import binary_metrics
from pcos_ecg.models.factory import make_pipeline
from pcos_ecg.synthesis.balance import make_balancer


def _toy(n_subj=30, n_pos=10, records=3, seed=0):
    """Repeated records per participant, so grouping actually matters."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        y = int(s < n_pos)
        centre = rng.normal(y, 1.0, 4)
        for r in range(records):
            rows.append(
                {"subject_id": f"S{s}", "record_id": f"S{s}-{r}", "label": y}
                | {f"f{j}": centre[j] + rng.normal(0, 0.3) for j in range(4)}
            )
    return pd.DataFrame(rows)


def _cfg(**over):
    cfg = {
        "subject_column": "subject_id",
        "label_column": "label",
        "aggregate_to_subject": None,
        "exclude_columns": ["record_id"],
        "feature_columns": None,
        "seed": 0,
        "cv": {"max_splits": 3, "min_subjects_per_class_per_fold": 2},
        "feature_selection": {"method": "mutual_info", "k": 2},
        "conditions": ["real_only", "smote"],
        "models": ["random_forest"],
        "smote": {"k_neighbors": 3, "min_minority_subjects": 4},
    }
    cfg.update(over)
    return cfg


class Recorder(BaseEstimator, TransformerMixin):
    """Pass-through step that logs which rows it was fitted on and asked to transform."""

    log: list = []

    def fit(self, X, y=None):
        Recorder.log.append(("fit", set(np.asarray(X)[:, 0].round(9))))
        return self

    def transform(self, X):
        Recorder.log.append(("transform", set(np.asarray(X)[:, 0].round(9))))
        return X


def test_no_participant_in_both_train_and_test():
    df = _toy()
    X, y, groups = prepare_xy(df, _cfg())
    res = run_experiment(X, y, groups, _cfg(conditions=["real_only"]))
    # Each participant contributes all 3 records to exactly one test fold.
    per_fold = res["per_fold"]
    assert per_fold["n_test_participants"].sum() == 30
    assert (per_fold["n_test_pos"] + per_fold["n_test_neg"]).sum() == 90


def test_preprocessing_selection_and_smote_fit_on_training_rows_only():
    df = _toy()
    X, y, groups = prepare_xy(df, _cfg())
    X = X.copy()
    X.insert(0, "tag", np.arange(len(X), dtype=float))  # unique per row
    Recorder.log = []

    def factory(*args):
        inner = make_pipeline(*args)
        return Pipeline([("rec", Recorder()), *inner.steps])

    run_experiment(X, y, groups, _cfg(), pipeline_factory=factory)
    # Log alternates: fit+transform on train (fit_transform) then transform on test.
    fits = [rows for kind, rows in Recorder.log if kind == "fit"]
    transforms = [rows for kind, rows in Recorder.log if kind == "transform"]
    test_sets = [t for t in transforms if t not in fits]
    assert len(fits) == len(test_sets) == 6  # 3 folds x 2 conditions
    for train_rows, test_rows in zip(fits, test_sets, strict=True):
        assert train_rows.isdisjoint(test_rows)
        train_subj = {groups[int(i)] for i in train_rows}
        test_subj = {groups[int(i)] for i in test_rows}
        assert train_subj.isdisjoint(test_subj)


def test_smote_adds_training_rows_but_test_counts_stay_real():
    df = _toy()
    cfg = _cfg(aggregate_to_subject="median")
    X, y, groups = prepare_xy(df, cfg)
    res = run_experiment(X, y, groups, cfg)
    pf = res["per_fold"]
    smote = pf[pf["condition"] == "smote"]
    assert (smote["n_synthetic_train_rows"] > 0).all()
    assert (pf["n_test_pos"] + pf["n_test_neg"] == pf["n_test_participants"]).all()
    assert (pf.loc[pf["condition"] == "real_only", "n_synthetic_train_rows"] == 0).all()


def test_smote_skipped_with_too_few_minority_participants():
    decision = make_balancer(n_minority_train=3, k_neighbors=3, min_minority=4, seed=0)
    assert decision.sampler is None and "skipped" in decision.skipped_reason
    assert make_balancer(5, 10, 4, 0).sampler.k_neighbors == 4


def test_tiny_cohort_is_not_estimable():
    y = np.array([1, 1, 0, 0, 0, 0])
    groups = np.array(list("abcdef"))
    with pytest.raises(NotEstimableError):
        choose_n_splits(y, groups, max_splits=5, min_per_fold=2)


def test_folds_limited_by_minority_participants_not_rows():
    y = np.array([1] * 6 + [0] * 30)
    groups = np.repeat(np.arange(12), 3)  # 2 positive participants with 3 rows each
    with pytest.raises(NotEstimableError):
        choose_n_splits(y, groups, max_splits=5, min_per_fold=2)


def test_single_class_fold_gives_nan_auc():
    m = binary_metrics(np.array([0, 0, 0]), np.array([0.1, 0.7, 0.2]))
    assert np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"]) and np.isnan(m["sensitivity"])
    assert m["specificity"] == pytest.approx(2 / 3)


def test_no_positive_predictions_gives_nan_ppv():
    m = binary_metrics(np.array([1, 0, 1, 0]), np.array([0.1, 0.2, 0.3, 0.4]))
    assert np.isnan(m["ppv"]) and m["sensitivity"] == 0 and m["specificity"] == 1


def test_identifiers_never_become_predictors():
    df = _toy()
    df["record_number"] = range(len(df))
    X, _, _ = prepare_xy(df, _cfg(exclude_columns=["record_id", "record_number"]))
    assert {"subject_id", "label", "record_number"}.isdisjoint(X.columns)
