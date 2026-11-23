"""Leakage-safe evaluation of the study's method.

What follows the original study (original_study/03_Code):
- Synthetic expansion towards 80 normal / 70 PCOS training rows (03_2).
- Correlation filter -> mutual information -> Boruta feature selection (03_3).
- Random Forest, XGBoost and SVM on ECG-only, physiology-only and combined features: the 9
  experiments of 03_4, plus the 0.7 ECG / 0.3 physiology ensemble.

What is done differently, on purpose:
- Real participants are split first. Synthetic rows are generated from the training
  participants of each fold only, and every test participant is a real, held-out person.
  The study trained and tested on synthetic rows.
- Feature selection runs inside each fold on its training rows only.
- Leave-one-subject-out when the cohort is small enough; otherwise participant-grouped,
  stratified k-fold. With leave-one-out each test fold has one person, so metrics are
  computed on the pooled out-of-fold predictions.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from pcos_ecg.evaluation.metrics import SUMMARY_METRICS, binary_metrics
from pcos_ecg.features.selection import select_features
from pcos_ecg.features.study import BINARY_CODES, feature_sets
from pcos_ecg.models.ensemble import WeightedSoftVote
from pcos_ecg.models.factory import CONDITIONS, make_pipeline
from pcos_ecg.synthesis.balance import make_balancer
from pcos_ecg.synthesis.generators import make_synthesizer, rows_needed


class NotEstimableError(ValueError):
    """The cohort is too small for an honest cross-validated comparison."""


def prepare_table(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Candidate features (every column named in any feature set), labels and participant IDs."""
    sid, label = cfg["subject_column"], cfg["label_column"]
    for col in (sid, label):
        if col not in df.columns:
            raise ValueError(f"Feature table lacks required column {col!r}")
    if "quality_ok" in df.columns:
        df = df[df["quality_ok"].astype(bool)]
    cols = list(dict.fromkeys(c for s in feature_sets(cfg).values() for c in s))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature table lacks configured features: {missing}")
    if cfg.get("aggregate_to_subject"):
        how = cfg["aggregate_to_subject"]
        df = df.groupby(sid).agg({**dict.fromkeys(cols, how), label: "first"}).reset_index()
    df = df.reset_index(drop=True)
    return df[cols].astype(float), df[label].to_numpy(int), df[sid].astype(str).to_numpy()


def make_splits(y: np.ndarray, groups: np.ndarray, cv_cfg: dict, seed: int):
    """Returns (strategy name, list of (train_idx, test_idx))."""
    per_person = pd.Series(y).groupby(groups).first()
    counts = per_person.value_counts()
    if len(counts) < 2:
        raise NotEstimableError("Only one class present among participants")
    n_people, n_minority = len(per_person), int(counts.min())
    strategy = cv_cfg.get("strategy", "auto")
    if strategy == "auto":
        strategy = "loso" if n_people <= cv_cfg.get("loso_max_participants", 60) else "group_kfold"
    if strategy == "loso":
        if n_minority < 2:
            raise NotEstimableError("Need at least 2 participants in each class")
        return "leave-one-subject-out", list(LeaveOneGroupOut().split(y, y, groups))
    if strategy != "group_kfold":
        raise ValueError(f"cv.strategy must be auto, loso or group_kfold, got {strategy!r}")
    k = min(
        cv_cfg.get("max_splits", 5),
        n_minority // max(cv_cfg.get("min_subjects_per_class_per_fold", 2), 1),
    )
    if k < 2:
        raise NotEstimableError(
            f"{n_minority} participant(s) in the smaller class cannot fill 2 folds; "
            "use cv.strategy: loso"
        )
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return f"participant-grouped stratified {k}-fold", list(splitter.split(y, y, groups))


def _expand(
    X_tr, y_tr, cfg, fold, synthesizer_factory, warn
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Study-style synthetic expansion of one training fold. Returns (X, y, n_synthetic)."""
    s = cfg.get("synthesis") or {}
    counts = rows_needed(y_tr, s.get("target_counts", {0: 80, 1: 70}))
    min_real = s.get("min_real_per_class", 3)
    small = [c for c in counts if (y_tr == c).sum() < min_real]
    if small:
        warn(f"fold {fold}: fewer than {min_real} real participants in class {small}; no synthesis")
        return X_tr, y_tr, 0
    categorical = [c for c in BINARY_CODES if c in X_tr.columns]
    gen = synthesizer_factory(
        s.get("method", "class_conditional_gaussian"),
        categorical,
        seed=int(cfg["seed"]) + fold,
        epochs=s.get("ctgan_epochs", 300),
    )
    X_syn, y_syn = gen.fit(X_tr, y_tr).sample(counts)
    X_all = pd.concat([X_tr, X_syn.astype(float)], ignore_index=True)
    return X_all, np.concatenate([y_tr, y_syn.astype(int)]), len(y_syn)


def run_experiment(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: dict,
    synthesizer_factory: Callable = make_synthesizer,
) -> dict:
    seed = int(cfg["seed"])
    strategy, folds = make_splits(y, groups, cfg.get("cv") or {}, seed)
    sets = feature_sets(cfg)
    fs_cfg = cfg.get("feature_selection") or {}
    conditions = cfg.get("conditions", ["study_synthesis"])
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown conditions {sorted(unknown)}; choose from {CONDITIONS}")
    models = cfg.get("models", ["random_forest", "xgboost", "svm"])
    ens = cfg.get("ensemble") or {}
    smote_cfg = cfg.get("smote") or {}

    warn_log: list[str] = []
    oof, fold_rows, selection_log = [], [], []
    for fold, (tr, te) in enumerate(folds):
        if set(groups[tr]) & set(groups[te]):
            raise RuntimeError("participant appears in both training and test fold")
        X_tr, y_tr, X_te = X.iloc[tr].reset_index(drop=True), y[tr], X.iloc[te]
        for condition in conditions:
            n_synth = 0
            if condition == "study_synthesis":
                X_fit, y_fit, n_synth = _expand(
                    X_tr, y_tr, cfg, fold, synthesizer_factory, warn_log.append
                )
            else:
                X_fit, y_fit = X_tr, y_tr
            chosen = {}
            for set_name, cols in sets.items():
                sel = select_features(
                    X_fit[cols],
                    y_fit,
                    method=fs_cfg.get("method", "study"),
                    corr_threshold=fs_cfg.get("corr_threshold", 0.90),
                    boruta_max_iter=fs_cfg.get("boruta_max_iter", 50),
                    fallback_k=fs_cfg.get("fallback_k", 4),
                    seed=seed,
                )
                chosen[set_name] = sel.selected
                selection_log.append(
                    {
                        "fold": fold,
                        "condition": condition,
                        "feature_set": set_name,
                        "selected": sel.selected,
                        "fallback_used": sel.fallback_used,
                    }
                )
                n_pos, n_neg = int((y_fit == 1).sum()), int((y_fit == 0).sum())
                for model in models:
                    sampler = None
                    if condition == "smote":
                        n_min = len(set(groups[tr][y_tr == (1 if n_pos <= n_neg else 0)]))
                        decision = make_balancer(
                            n_min,
                            smote_cfg.get("k_neighbors", 3),
                            smote_cfg.get("min_minority_subjects", 4),
                            seed,
                        )
                        sampler = decision.sampler
                        if decision.skipped_reason:
                            warn_log.append(
                                f"fold {fold} {model}/{set_name}: {decision.skipped_reason}"
                            )
                    pipe = make_pipeline(model, condition, sampler, n_pos, n_neg, seed)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        pipe.fit(X_fit[sel.selected], y_fit)
                    score = pipe.predict_proba(X_te[sel.selected])[:, 1]
                    oof.append(_rows(te, y[te], score, fold, model, set_name, condition, n_synth))
            if ens.get("enabled") and condition == "study_synthesis":
                ecg, phys = chosen.get("ECG Only"), chosen.get("Physiology Only")
                if ecg and phys:
                    ensemble = WeightedSoftVote(
                        ecg, phys, tuple(ens.get("weights", (0.7, 0.3))), seed
                    )
                    ensemble.fit(X_fit, y_fit)
                    score = ensemble.predict_proba(X_te)[:, 1]
                    oof.append(
                        _rows(
                            te,
                            y[te],
                            score,
                            fold,
                            "ensemble",
                            "ECG + Physiology",
                            condition,
                            n_synth,
                        )
                    )
        fold_rows.append(
            {
                "fold": fold,
                "n_train_participants": len(set(groups[tr])),
                "n_test_participants": len(set(groups[te])),
                "n_test_pos": int((y[te] == 1).sum()),
                "n_test_neg": int((y[te] == 0).sum()),
            }
        )

    oof_df = pd.concat(oof, ignore_index=True)
    return {
        "strategy": strategy,
        "n_splits": len(folds),
        "n_participants": int(len(set(groups))),
        "n_positive_participants": int(pd.Series(y).groupby(groups).first().sum()),
        "oof": oof_df,
        "summary": summarise(oof_df, per_fold=len(folds) <= 20),
        "folds": pd.DataFrame(fold_rows),
        "selection_frequency": _selection_frequency(selection_log, len(folds)),
        "fallback_count": int(sum(s["fallback_used"] for s in selection_log)),
        "warnings": sorted(set(warn_log)),
    }


def _rows(te, y_true, score, fold, model, feature_set, condition, n_synth) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row": te,
            "y_true": y_true,
            "score": score,
            "fold": fold,
            "model": model,
            "feature_set": feature_set,
            "condition": condition,
            "n_synthetic_train_rows": n_synth,
        }
    )


def summarise(oof: pd.DataFrame, per_fold: bool = True) -> pd.DataFrame:
    """Metrics on pooled out-of-fold predictions; plus mean/SD over folds where folds are big."""
    rows = []
    keys = ["condition", "model", "feature_set"]
    for key, part in oof.groupby(keys, sort=False):
        m = binary_metrics(part["y_true"].to_numpy(), part["score"].to_numpy())
        row = dict(zip(keys, key, strict=True)) | {
            k: m[k] for k in ("tp", "tn", "fp", "fn", *SUMMARY_METRICS)
        }
        if per_fold:
            fold_auc = [
                binary_metrics(f["y_true"].to_numpy(), f["score"].to_numpy())["roc_auc"]
                for _, f in part.groupby("fold")
            ]
            row["roc_auc_fold_sd"] = (
                float(np.nanstd(fold_auc, ddof=1)) if len(fold_auc) > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def _selection_frequency(log: list[dict], n_folds: int) -> pd.DataFrame:
    counts: Counter = Counter()
    for entry in log:
        for feat in entry["selected"]:
            counts[(entry["condition"], entry["feature_set"], feat)] += 1
    rows = [
        {
            "condition": c,
            "feature_set": s,
            "feature": f,
            "folds_selected": n,
            "share": round(n / n_folds, 3),
        }
        for (c, s, f), n in counts.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["condition", "feature_set", "share"], ascending=[True, True, False]
    )
