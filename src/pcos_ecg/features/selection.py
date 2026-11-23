"""The study's three-stage feature selection (original_study/03_Code/03_3):
correlation filter (|r| > 0.90) -> mutual information -> Boruta.

Difference from the study: the study ran this once on the whole 150-row dataset before
cross-validation, so the test rows influenced which features were kept. Here
`select_features` is called inside each fold on that fold's training rows only.

Two small, documented choices the notebook left open:
- In a correlated pair, the feature with the lower mutual information is dropped.
- Boruta "Confirmed" and "Tentative" features are both kept (the study's final rerun did the
  same). If Boruta keeps nothing, which can happen with few training participants, the top
  `fallback_k` features by mutual information are kept and the fallback is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif


@dataclass
class SelectionResult:
    selected: list[str]
    dropped_empty: list[str] = field(default_factory=list)
    dropped_correlated: dict[str, str] = field(default_factory=dict)  # dropped -> kept partner
    mutual_information: dict[str, float] = field(default_factory=dict)
    boruta_status: dict[str, str] = field(default_factory=dict)
    fallback_used: bool = False


def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    method: str = "study",
    corr_threshold: float = 0.90,
    boruta_max_iter: int = 50,
    fallback_k: int = 4,
    seed: int = 42,
) -> SelectionResult:
    """method: "study" (corr -> MI -> Boruta), "mutual_info" (corr -> top-k MI) or "none"."""
    y = np.asarray(y)
    empty = [c for c in X.columns if X[c].isna().all()]
    X = X.drop(columns=empty)
    X = X.fillna(X.median())
    if method == "none" or X.shape[1] == 0:
        return SelectionResult(list(X.columns), dropped_empty=empty)

    mi = pd.Series(mutual_info_classif(X, y, random_state=seed), index=X.columns)

    # Stage 1: drop the less informative member of each highly correlated pair.
    corr = X.corr().abs()
    dropped: dict[str, str] = {}
    for i, a in enumerate(X.columns):
        for b in X.columns[i + 1 :]:
            if a in dropped or b in dropped or not corr.loc[a, b] > corr_threshold:
                continue
            loser, keeper = (a, b) if mi[a] < mi[b] else (b, a)
            dropped[loser] = keeper
    kept = [c for c in X.columns if c not in dropped]

    result = SelectionResult(
        selected=[],
        dropped_empty=empty,
        dropped_correlated=dropped,
        mutual_information={c: float(mi[c]) for c in X.columns},
    )
    top_by_mi = list(mi[kept].sort_values(ascending=False).index)

    if method == "mutual_info":
        result.selected = top_by_mi[:fallback_k]
        return result
    if method != "study":
        raise ValueError(
            f"selection method must be 'study', 'mutual_info' or 'none', got {method!r}"
        )

    # Stage 3: Boruta, as configured in the study's notebook.
    try:
        from boruta import BorutaPy
    except ImportError as exc:
        raise ImportError(
            "Boruta selection needs the 'study' extra: pip install -e '.[study]'"
        ) from exc
    forest = RandomForestClassifier(
        n_jobs=1, class_weight="balanced", max_depth=5, random_state=seed
    )
    boruta = BorutaPy(forest, n_estimators="auto", max_iter=boruta_max_iter, random_state=seed)
    boruta.fit(X[kept].to_numpy(), y)
    status = np.where(
        boruta.support_, "Confirmed", np.where(boruta.support_weak_, "Tentative", "Rejected")
    )
    result.boruta_status = dict(zip(kept, status, strict=True))
    result.selected = [c for c, s in result.boruta_status.items() if s != "Rejected"]
    if not result.selected:
        result.selected = top_by_mi[:fallback_k]
        result.fallback_used = True
    return result
