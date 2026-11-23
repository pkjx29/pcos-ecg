"""The study's ECG-weighted soft-vote ensemble (original_study/03_Code/03_3, step 4).

Two Random Forests, one on ECG features and one on physiological features, combined as
0.7 * P(PCOS | ECG) + 0.3 * P(PCOS | physiology). The weights come from the study; they were
chosen to favour the ECG, not tuned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


class WeightedSoftVote(ClassifierMixin, BaseEstimator):
    def __init__(self, ecg_features=(), physio_features=(), weights=(0.7, 0.3), seed=42):
        self.ecg_features = ecg_features
        self.physio_features = physio_features
        self.weights = weights
        self.seed = seed

    def _member(self):
        forest = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=self.seed, n_jobs=1
        )
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), forest)

    def fit(self, X: pd.DataFrame, y):
        if not self.ecg_features or not self.physio_features:
            raise ValueError("both ECG and physiological feature lists are needed")
        if abs(sum(self.weights) - 1.0) > 1e-9:
            raise ValueError("weights must sum to 1")
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.ecg_model_ = self._member().fit(X[list(self.ecg_features)], y)
        self.physio_model_ = self._member().fit(X[list(self.physio_features)], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        w_ecg, w_phys = self.weights
        p = w_ecg * self.ecg_model_.predict_proba(X[list(self.ecg_features)])[:, 1]
        p += w_phys * self.physio_model_.predict_proba(X[list(self.physio_features)])[:, 1]
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
