"""Model pipelines. Everything that learns from data (imputer, scaler, feature selector,
SMOTE, classifier) lives inside one imblearn Pipeline, so cross-validation refits all of
it on each training fold and nothing is fitted on test rows.

Hyperparameters are fixed, defensible defaults: on a small cohort a search would overfit
the validation folds.
"""

from __future__ import annotations

from functools import partial

from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODELS = ("baseline", "random_forest", "svm", "xgboost")
CONDITIONS = ("real_only", "class_weight", "smote")


def make_classifier(name: str, condition: str, n_pos: int, n_neg: int, seed: int):
    weighted = condition == "class_weight"
    if name == "baseline":
        # Predicts the training prevalence; anything useful must beat this.
        return DummyClassifier(strategy="prior")
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced" if weighted else None,
            random_state=seed,
            n_jobs=1,
        )
    if name == "svm":
        # Platt scaling via CalibratedClassifierCV, as scikit-learn now recommends instead of
        # SVC(probability=True). ensemble=False keeps one SVM fitted on the whole training fold.
        svc = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced" if weighted else None,
            random_state=seed,
        )
        return CalibratedClassifierCV(svc, method="sigmoid", cv=3, ensemble=False)
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(n_neg / max(n_pos, 1)) if weighted else 1.0,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {MODELS}")


def make_pipeline(
    model: str,
    condition: str,
    n_features: int,
    k_best: int | None,
    sampler,
    n_pos: int,
    n_neg: int,
    seed: int,
) -> Pipeline:
    """sampler: a fitted-per-fold SMOTE (or None). k_best: None disables selection."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; choose from {CONDITIONS}")
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if k_best is not None and k_best < n_features:
        score = partial(mutual_info_classif, random_state=seed)
        steps.append(("select", SelectKBest(score, k=k_best)))
    if sampler is not None:
        steps.append(("balance", sampler))
    steps.append(("clf", make_classifier(model, condition, n_pos, n_neg, seed)))
    return Pipeline(steps)
