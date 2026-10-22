"""Participant-aware, stratified cross-validation for the model comparison.

- One participant's rows never appear on both sides of a split (StratifiedGroupKFold).
- The number of folds is set by how many real minority-class participants there are,
  not by record or synthetic-row counts.
- Every model and condition is scored on the same folds.
- With so few participants, cross-validated scores are internal estimates only; they
  are not an external validation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from pcos_ecg.evaluation.metrics import SUMMARY_METRICS, binary_metrics
from pcos_ecg.models.factory import make_pipeline
from pcos_ecg.synthesis.balance import make_balancer


class NotEstimableError(ValueError):
    """The cohort is too small for an honest cross-validated comparison."""


def prepare_xy(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Select predictor columns and optionally collapse records to one row per participant."""
    sid, label = cfg["subject_column"], cfg["label_column"]
    for col in (sid, label):
        if col not in df.columns:
            raise ValueError(f"Feature table lacks required column {col!r}")
    if "quality_ok" in df.columns:
        df = df[df["quality_ok"].astype(bool)]

    excluded = set(cfg.get("exclude_columns") or []) | {sid, label}
    if cfg.get("feature_columns"):
        cols = list(cfg["feature_columns"])
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Configured feature columns not found: {sorted(missing)}")
    else:
        cols = [c for c in df.select_dtypes("number").columns if c not in excluded]
    if not cols:
        raise ValueError("No numeric predictor columns left after exclusions")

    if cfg.get("aggregate_to_subject"):
        agg = df.groupby(sid).agg(
            {**dict.fromkeys(cols, cfg["aggregate_to_subject"]), label: "first"}
        )
        df = agg.reset_index()
    X = df[cols].astype(float)
    keep = X.notna().any(axis=1)
    return (
        X[keep].reset_index(drop=True),
        df.loc[keep, label].to_numpy(int),
        df.loc[keep, sid].astype(str).to_numpy(),
    )


def choose_n_splits(y: np.ndarray, groups: np.ndarray, max_splits: int, min_per_fold: int) -> int:
    per_class = pd.Series(y).groupby(groups).first().value_counts()
    if len(per_class) < 2:
        raise NotEstimableError("Only one class present among participants")
    n_minority = int(per_class.min())
    n = min(max_splits, n_minority // max(min_per_fold, 1))
    if n < 2:
        raise NotEstimableError(
            f"{n_minority} participant(s) in the smaller class cannot fill 2 folds with "
            f"{min_per_fold} each; the comparison is not reliably estimable"
        )
    return n


PipelineFactory = Callable[..., object]


def run_experiment(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: dict,
    pipeline_factory: PipelineFactory = make_pipeline,
) -> dict:
    seed = int(cfg["seed"])
    cv_cfg, smote_cfg = cfg["cv"], cfg.get("smote", {})
    n_splits = choose_n_splits(
        y, groups, cv_cfg["max_splits"], cv_cfg["min_subjects_per_class_per_fold"]
    )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(X, y, groups))

    fs_cfg = cfg.get("feature_selection") or {}
    k_best = fs_cfg.get("k") if fs_cfg.get("method") == "mutual_info" else None

    per_fold, warnings = [], []
    selected: dict[str, Counter] = {}
    for model in cfg["models"]:
        for condition in cfg["conditions"]:
            key = f"{model}|{condition}"
            selected[key] = Counter()
            for i, (tr, te) in enumerate(folds):
                if set(groups[tr]) & set(groups[te]):
                    raise RuntimeError("participant appears in both training and test fold")
                y_tr = y[tr]
                n_pos, n_neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
                minority = 1 if n_pos <= n_neg else 0
                n_minority_subj = len(set(groups[tr][y_tr == minority]))

                sampler, skipped = None, None
                if condition == "smote":
                    decision = make_balancer(
                        n_minority_subj,
                        smote_cfg.get("k_neighbors", 3),
                        smote_cfg.get("min_minority_subjects", 4),
                        seed,
                    )
                    sampler, skipped = decision.sampler, decision.skipped_reason
                    if skipped:
                        warnings.append(f"{key} fold {i}: {skipped}")

                pipe = pipeline_factory(
                    model, condition, X.shape[1], k_best, sampler, n_pos, n_neg, seed
                )
                pipe.fit(X.iloc[tr], y_tr)
                score = pipe.predict_proba(X.iloc[te])[:, 1]

                n_synth = 0
                if sampler is not None and hasattr(sampler, "sampling_strategy_"):
                    n_synth = int(sum(sampler.sampling_strategy_.values()))
                if "select" in pipe.named_steps:
                    mask = pipe.named_steps["select"].get_support()
                    selected[key].update(X.columns[mask])

                per_fold.append(
                    {
                        "model": model,
                        "condition": condition,
                        "fold": i,
                        "n_train_participants": len(set(groups[tr])),
                        "n_test_participants": len(set(groups[te])),
                        "n_train_pos": n_pos,
                        "n_train_neg": n_neg,
                        "n_test_pos": int((y[te] == 1).sum()),
                        "n_test_neg": int((y[te] == 0).sum()),
                        "n_synthetic_train_rows": n_synth,
                        "smote_skipped": skipped is not None,
                        **binary_metrics(y[te], score),
                    }
                )

    fold_df = pd.DataFrame(per_fold)
    summary = (
        fold_df.groupby(["model", "condition"])[list(SUMMARY_METRICS)].agg(["mean", "std"]).round(3)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    return {
        "n_splits": n_splits,
        "split_strategy": "StratifiedGroupKFold (participant-grouped, stratified, shuffled)",
        "n_participants": int(len(set(groups))),
        "n_positive_participants": int(pd.Series(y).groupby(groups).first().sum()),
        "per_fold": fold_df,
        "summary": summary,
        "feature_selection_counts": {k: dict(v) for k, v in selected.items() if v},
        "warnings": warnings,
    }
