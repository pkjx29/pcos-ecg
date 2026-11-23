"""Synthetic participants, following the original study (original_study/03_Code).

Three generators share one interface: fit(features, labels), then sample({label: n_rows}).

- "class_conditional_gaussian": the study's 30 -> 150 method (03_2). The notebook calls it a
  "Bayesian network", but it fits, per class, a mean vector and a shrunk covariance
  (0.7 * covariance + 0.3 * its diagonal) and samples from that multivariate normal;
  binary answers are drawn from per-class frequencies. Two changes from 03_2: class sizes
  are fixed exactly instead of drawn at random, and numeric values are clipped to the
  training range so no negative durations appear.
- "gaussian_copula" and "ctgan": SDV synthesizers, as compared in the study's 03_1.

A true Bayesian-network synthesizer is not offered: SDV's free edition has none, and adding
one would need another library (pgmpy).

In evaluation these are only ever fitted on the training participants of a fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ("class_conditional_gaussian", "gaussian_copula", "ctgan")
SDV_METHODS = ("gaussian_copula", "ctgan")
LABEL = "__label__"


class ClassConditionalGaussian:
    def __init__(self, categorical: list[str] | None = None, seed: int = 42, clip: bool = True):
        self.categorical = list(categorical or [])
        self.seed = seed
        self.clip = clip

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> ClassConditionalGaussian:
        self.columns = list(X.columns)
        self.numeric = [c for c in self.columns if c not in self.categorical]
        filled = X.fillna(X.median(numeric_only=True))
        self.low, self.high = filled[self.numeric].min(), filled[self.numeric].max()
        y = np.asarray(y)
        self.params = {}
        for cls in np.unique(y):
            part = filled[y == cls]
            num = part[self.numeric].to_numpy(dtype=float)
            if len(part) >= 2:
                cov = np.cov(num, rowvar=False)
                cov = np.atleast_2d(cov)
                cov = 0.7 * cov + 0.3 * np.diag(np.diag(cov)) + np.eye(len(self.numeric)) * 1e-6
            else:
                cov = np.eye(len(self.numeric)) * 1e-4
            freqs = {c: part[c].value_counts(normalize=True) for c in self.categorical}
            self.params[cls] = (num.mean(axis=0), cov, freqs)
        return self

    def sample(self, counts: dict, seed: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        frames, labels = [], []
        for cls, n in counts.items():
            if n <= 0:
                continue
            if cls not in self.params:
                raise ValueError(f"class {cls!r} was not present when fitting")
            mean, cov, freqs = self.params[cls]
            block = pd.DataFrame(rng.multivariate_normal(mean, cov, size=n), columns=self.numeric)
            if self.clip:
                block = block.clip(self.low, self.high, axis=1)
            for col, f in freqs.items():
                block[col] = rng.choice(f.index.to_numpy(), size=n, p=f.to_numpy())
            frames.append(block[self.columns])
            labels.append(np.full(n, cls))
        if not frames:
            return pd.DataFrame(columns=self.columns), np.array([], dtype=int)
        return pd.concat(frames, ignore_index=True), np.concatenate(labels)


class SDVSynthesizer:
    """SDV Gaussian Copula or CTGAN, with the label modelled as a column and rows kept per class."""

    def __init__(
        self, kind: str, categorical: list[str] | None = None, seed: int = 42, epochs: int = 300
    ):
        if kind not in SDV_METHODS:
            raise ValueError(f"kind must be one of {SDV_METHODS}")
        self.kind, self.categorical, self.seed, self.epochs = (
            kind,
            list(categorical or []),
            seed,
            epochs,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> SDVSynthesizer:
        try:
            from sdv.metadata import Metadata
            from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
        except ImportError as exc:
            raise ImportError(
                "SDV synthesizers need the 'study' extra: pip install -e '.[study]'"
            ) from exc
        self.columns = list(X.columns)
        data = X.fillna(X.median(numeric_only=True)).assign(**{LABEL: np.asarray(y)})
        md = Metadata.detect_from_dataframes({"t": data})
        for col in self.categorical + [LABEL]:
            md.update_column(column_name=col, sdtype="categorical", table_name="t")
        for col in set(self.columns) - set(self.categorical):
            md.update_column(column_name=col, sdtype="numerical", table_name="t")
        if self.kind == "ctgan":
            self.model = CTGANSynthesizer(
                md, epochs=self.epochs, enforce_min_max_values=True, verbose=False
            )
        else:
            self.model = GaussianCopulaSynthesizer(md, enforce_min_max_values=True)
        _seed_everything(self.seed)
        self.model.fit({"t": data})
        return self

    def sample(
        self, counts: dict, seed: int | None = None, max_batches: int = 50
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Draw plain batches and keep rows per class until each count is met.

        SDV's conditional sampling rejects rows one batch at a time and can take minutes for
        CTGAN on a tiny table; filling classes from ordinary batches is equivalent and fast.
        """
        _seed_everything(self.seed if seed is None else seed)
        need = {cls: int(n) for cls, n in counts.items() if n > 0}
        if not need:
            return pd.DataFrame(columns=self.columns), np.array([], dtype=int)
        batch = max(200, 2 * sum(need.values()))
        kept: dict = {cls: [] for cls in need}
        for _ in range(max_batches):
            out = self.model.sample(batch)
            out = out["t"] if isinstance(out, dict) else out
            for cls in need:
                have = sum(len(f) for f in kept[cls])
                if have < need[cls]:
                    kept[cls].append(out[out[LABEL] == cls].head(need[cls] - have))
            if all(sum(len(f) for f in kept[c]) >= n for c, n in need.items()):
                break
        else:
            short = {c: n - sum(len(f) for f in kept[c]) for c, n in need.items()}
            raise RuntimeError(f"{self.kind} produced too few rows for some classes: {short}")
        rows = pd.concat([f for parts in kept.values() for f in parts], ignore_index=True)
        return rows[self.columns], rows[LABEL].to_numpy()


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


class _SkipEmptyColumns:
    """Leaves columns with no values out of fitting and returns them empty in samples, so a
    feature that could not be measured stays missing instead of breaking the fit."""

    def __init__(self, inner):
        self.inner = inner

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.columns = list(X.columns)
        self.empty = [c for c in X.columns if X[c].isna().all()]
        self.inner.fit(X.drop(columns=self.empty), y)
        return self

    def sample(self, counts: dict, seed: int | None = None):
        Xs, ys = self.inner.sample(counts, seed=seed)
        for col in self.empty:
            Xs[col] = np.nan
        return Xs[self.columns], ys


def make_synthesizer(method: str, categorical: list[str] | None = None, seed: int = 42, **opts):
    if method == "class_conditional_gaussian":
        inner = ClassConditionalGaussian(categorical, seed=seed, clip=opts.get("clip", True))
    elif method in SDV_METHODS:
        inner = SDVSynthesizer(method, categorical, seed=seed, epochs=opts.get("epochs", 300))
    else:
        raise ValueError(f"Unknown synthesis method {method!r}; choose from {METHODS}")
    return _SkipEmptyColumns(inner)


def rows_needed(y_train: np.ndarray, target_counts: dict) -> dict:
    """Synthetic rows per class so that real + synthetic reaches the target (never negative)."""
    y_train = np.asarray(y_train)
    return {
        int(c): max(0, int(n) - int((y_train == int(c)).sum())) for c, n in target_counts.items()
    }
