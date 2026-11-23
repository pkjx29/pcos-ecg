"""Model pipelines. Imputation, scaling, SMOTE (baseline only) and the classifier sit in one
imblearn Pipeline that is refitted on each training fold, so nothing is fitted on test rows.
Feature selection and synthetic expansion happen just before, on the same training fold
(evaluation/cv.py).

Hyperparameters are fixed defaults close to the study's notebooks (03_4): a search on a
cohort of about 30 people would overfit the validation folds.
"""

from __future__ import annotations

from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODELS = ("random_forest", "xgboost", "svm", "baseline")
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "svm": "SVM (RBF)",
    "baseline": "Baseline (prior)",
    "ensemble": "Ensemble 0.7 ECG / 0.3 physio",
}
# The study's approach first; the rest are labelled baselines.
CONDITIONS = ("study_synthesis", "real_only", "class_weight", "smote")
CONDITION_LABELS = {
    "study_synthesis": "Study method: synthetic expansion of training participants",
    "real_only": "Baseline: real participants only",
    "class_weight": "Baseline: class weights",
    "smote": "Baseline: SMOTE on training participants",
}


def make_classifier(name: str, condition: str, n_pos: int, n_neg: int, seed: int):
    weighted = condition == "class_weight"
    if name == "baseline":
        # Predicts the training prevalence; anything useful must beat this.
        return DummyClassifier(strategy="prior")
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
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
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            scale_pos_weight=(n_neg / max(n_pos, 1)) if weighted else 1.0,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {MODELS}")


def make_pipeline(model: str, condition: str, sampler, n_pos: int, n_neg: int, seed: int):
    """sampler: a SMOTE fitted inside this pipeline (smote baseline only), or None."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; choose from {CONDITIONS}")
    steps = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ]
    if sampler is not None:
        steps.append(("balance", sampler))
    steps.append(("clf", make_classifier(model, condition, n_pos, n_neg, seed)))
    return Pipeline(steps)
