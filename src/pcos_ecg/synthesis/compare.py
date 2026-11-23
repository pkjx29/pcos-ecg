"""Compare synthesizers the way the study's 03_1 notebook did.

Each method is fitted on the same table and asked for the same number of rows. Scores:
SDMetrics overall quality (column shapes and column-pair trends), mean absolute difference
between the real and synthetic Pearson correlation matrices, and mean KS statistic and mean
Wasserstein distance over numeric columns. Ranking follows 03_1: highest quality, then
lowest correlation difference, lowest KS, lowest Wasserstein.

This is a fidelity check of the generators, not a model evaluation; it is run on the whole
table and its result only chooses which generator the evaluation uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from pcos_ecg.synthesis.generators import make_synthesizer


def _quality(real: pd.DataFrame, synth: pd.DataFrame, categorical: list[str]) -> dict[str, float]:
    try:
        from sdmetrics.reports import QualityReport
    except ImportError:
        return dict.fromkeys(["Overall Quality", "Column Shapes", "Column Pair Trends"], np.nan)
    columns = {
        c: {"sdtype": "categorical" if c in categorical else "numerical"} for c in real.columns
    }
    report = QualityReport()
    # sdmetrics' current report takes {table: frame} and multi-table metadata.
    report.generate(
        {"t": real}, {"t": synth}, {"tables": {"t": {"columns": columns}}}, verbose=False
    )
    props = dict(
        zip(report.get_properties()["Property"], report.get_properties()["Score"], strict=True)
    )
    return {
        "Overall Quality": float(report.get_score()),
        "Column Shapes": float(props.get("Column Shapes", np.nan)),
        "Column Pair Trends": float(props.get("Column Pair Trends", np.nan)),
    }


def compare_synthesizers(
    X: pd.DataFrame,
    y: np.ndarray,
    methods: list[str],
    n_rows: int,
    categorical: list[str],
    seed: int = 42,
    epochs: int = 300,
) -> pd.DataFrame:
    y = np.asarray(y)
    real = X.fillna(X.median(numeric_only=True)).assign(label=y)
    numeric = [c for c in X.columns if c not in categorical]
    shares = pd.Series(y).value_counts(normalize=True)
    counts = {int(c): int(round(n_rows * s)) for c, s in shares.items()}
    real_corr = real[numeric].corr()

    rows = []
    for method in methods:
        synth_X, synth_y = (
            make_synthesizer(method, categorical, seed=seed, epochs=epochs).fit(X, y).sample(counts)
        )
        synth = synth_X.assign(label=synth_y)
        diff = (real_corr - synth[numeric].astype(float).corr()).abs().to_numpy()
        upper = diff[np.triu_indices_from(diff, k=1)]
        rows.append(
            {
                "Synthesizer": method,
                **_quality(real, synth, categorical + ["label"]),
                "Mean Correlation Difference": float(np.nanmean(upper)),
                "Mean KS Statistic": float(
                    np.mean([ks_2samp(real[c], synth[c].astype(float)).statistic for c in numeric])
                ),
                "Mean Wasserstein Distance": float(
                    np.mean(
                        [wasserstein_distance(real[c], synth[c].astype(float)) for c in numeric]
                    )
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        [
            "Overall Quality",
            "Mean Correlation Difference",
            "Mean KS Statistic",
            "Mean Wasserstein Distance",
        ],
        ascending=[False, True, True, True],
        na_position="last",
    )
    table.insert(0, "Rank", np.arange(1, len(table) + 1))
    return table.reset_index(drop=True)
