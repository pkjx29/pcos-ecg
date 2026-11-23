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
    """SDV Gaussian Copula or CTGAN, sampled per class with SDV's conditional sampling."""

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

    def sample(self, counts: dict, seed: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
        from sdv.sampling import Condition

        _seed_everything(self.seed if seed is None else seed)
        conditions = [
            Condition(num_rows=int(n), column_values={LABEL: cls})
            for cls, n in counts.items()
            if n > 0
        ]
        if not conditions:
            return pd.DataFrame(columns=self.columns), np.array([], dtype=int)
        out = self.model.sample_from_conditions(conditions)
        out = out["t"] if isinstance(out, dict) else out
        return out[self.columns].reset_index(drop=True), out[LABEL].to_numpy()


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def make_synthesizer(method: str, categorical: list[str] | None = None, seed: int = 42, **opts):
    if method == "class_conditional_gaussian":
        return ClassConditionalGaussian(categorical, seed=seed, clip=opts.get("clip", True))
    if method in SDV_METHODS:
        return SDVSynthesizer(method, categorical, seed=seed, epochs=opts.get("epochs", 300))
    raise ValueError(f"Unknown synthesis method {method!r}; choose from {METHODS}")


def rows_needed(y_train: np.ndarray, target_counts: dict) -> dict:
    """Synthetic rows per class so that real + synthetic reaches the target (never negative)."""
    y_train = np.asarray(y_train)
    return {
        int(c): max(0, int(n) - int((y_train == int(c)).sum())) for c, n in target_counts.items()
    }
