"""The study's analysis outputs (03_3 and 03_4), written to the run folder.

- Calibration and decision curves use the out-of-fold predictions for held-out real
  participants, so they describe the evaluated models.
- Mann-Whitney U tests compare the real participants' features between classes.
- The learning curve refits a Random Forest on growing subsets of real participants.
- SHAP refits Random Forest and XGBoost once on all participants (expanded as in the study)
  purely to explain them; those fits are not used for any metric.
- PCA, t-SNE and UMAP plot one point per real participant. For study data these plots are
  private: they stay in runs/, which is git-ignored.

Each step is skipped, with a note, if its optional library is missing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.model_selection import StratifiedKFold, learning_curve  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from pcos_ecg.features.study import BINARY_CODES  # noqa: E402
from pcos_ecg.models.factory import MODEL_LABELS  # noqa: E402
from pcos_ecg.synthesis.generators import make_synthesizer, rows_needed  # noqa: E402

PRIMARY = "study_synthesis"


def mann_whitney(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        a, b = X.loc[y == 0, col].dropna(), X.loc[y == 1, col].dropna()
        if len(a) < 2 or len(b) < 2 or pd.concat([a, b]).nunique() < 2:
            rows.append({"feature": col, "U": np.nan, "p_value": np.nan, "note": "too few values"})
            continue
        res = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {"feature": col, "U": float(res.statistic), "p_value": float(res.pvalue), "note": ""}
        )
    return pd.DataFrame(rows).sort_values("p_value", na_position="last")


def net_benefit(y_true: np.ndarray, score: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = len(y_true)
    out = []
    for t in thresholds:
        tp = np.sum((score >= t) & (y_true == 1))
        fp = np.sum((score >= t) & (y_true == 0))
        out.append(tp / n - fp / n * (t / (1 - t)))
    return np.array(out)


def _curves(oof: pd.DataFrame, out: Path) -> list[str]:
    main = oof[oof["condition"] == PRIMARY]
    if main.empty:
        return ["Calibration and decision curves skipped: the study condition was not run"]
    thresholds = np.linspace(0.01, 0.99, 99)
    cal_rows, dca_rows = [], []
    fig_c, ax_c = plt.subplots(figsize=(6, 6))
    fig_d, ax_d = plt.subplots(figsize=(7, 5))
    for (model, fset), part in main.groupby(["model", "feature_set"], sort=False):
        y_t, s = part["y_true"].to_numpy(), part["score"].to_numpy()
        label = f"{MODEL_LABELS.get(model, model)} / {fset}"
        if len(np.unique(y_t)) == 2:
            prob_true, prob_pred = calibration_curve(y_t, s, n_bins=5, strategy="quantile")
            cal_rows += [
                {"model": model, "feature_set": fset, "mean_predicted": p, "observed_rate": t}
                for p, t in zip(prob_pred, prob_true, strict=True)
            ]
            ax_c.plot(prob_pred, prob_true, "o-", label=label, alpha=0.8)
        nb = net_benefit(y_t, s, thresholds)
        dca_rows += [
            {"model": model, "feature_set": fset, "threshold": t, "net_benefit": b}
            for t, b in zip(thresholds, nb, strict=True)
        ]
        if model in ("random_forest", "ensemble"):
            ax_d.plot(thresholds, nb, label=label)
    y_all = main.drop_duplicates("row")["y_true"].to_numpy()
    ax_d.plot(
        thresholds, net_benefit(y_all, np.ones(len(y_all)), thresholds), "k--", label="Treat all"
    )
    ax_d.axhline(0, color="grey", lw=1, label="Treat none")
    ax_d.set(
        xlabel="Threshold probability",
        ylabel="Net benefit",
        ylim=(-0.2, 0.6),
        title="Decision curves",
    )
    ax_d.legend(fontsize=7)
    ax_c.plot([0, 1], [0, 1], "k--", lw=1)
    ax_c.set(xlabel="Mean predicted probability", ylabel="Observed PCOS rate", title="Calibration")
    ax_c.legend(fontsize=6)
    fig_c.savefig(out / "calibration_curves.png", dpi=150, bbox_inches="tight")
    fig_d.savefig(out / "decision_curves.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    pd.DataFrame(cal_rows).to_csv(out / "calibration_curves.csv", index=False)
    pd.DataFrame(dca_rows).to_csv(out / "decision_curves.csv", index=False)
    return [
        "calibration_curves.png/.csv and decision_curves.png/.csv: "
        "out-of-fold, held-out real participants"
    ]


def _learning_curve(X: pd.DataFrame, y: np.ndarray, seed: int, out: Path) -> list[str]:
    k = min(5, int(np.bincount(y).min()))
    if k < 2:
        return ["Learning curve skipped: too few participants per class"]
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
    )
    sizes, train, test = learning_curve(
        model,
        X,
        y,
        cv=StratifiedKFold(k, shuffle=True, random_state=seed),
        scoring="roc_auc",
        train_sizes=np.linspace(0.4, 1.0, 4),
        random_state=seed,
    )
    pd.DataFrame(
        {"train_participants": sizes, "train_auc": train.mean(1), "cv_auc": np.nanmean(test, 1)}
    ).to_csv(out / "learning_curve.csv", index=False)
    plt.figure(figsize=(6, 4))
    plt.plot(sizes, train.mean(1), "o-", label="training")
    plt.plot(sizes, np.nanmean(test, 1), "s-", label="cross-validated")
    plt.xlabel("Real training participants")
    plt.ylabel("ROC-AUC")
    plt.title("Learning curve (Random Forest, all features, no synthesis)")
    plt.legend()
    plt.savefig(out / "learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    return ["learning_curve.png/.csv: Random Forest on real participants only"]


def _shap(X: pd.DataFrame, y: np.ndarray, cfg: dict, out: Path) -> list[str]:
    try:
        import shap
    except ImportError:
        return ["SHAP skipped: install the 'study' extra"]
    seed = int(cfg["seed"])
    s = cfg.get("synthesis") or {}
    categorical = [c for c in BINARY_CODES if c in X.columns]
    gen = make_synthesizer(s.get("method", "class_conditional_gaussian"), categorical, seed=seed)
    Xs, ys = gen.fit(X, y).sample(rows_needed(y, s.get("target_counts", {0: 80, 1: 70})))
    X_fit = pd.concat([X, Xs.astype(float)], ignore_index=True).fillna(X.median())
    y_fit = np.concatenate([y, ys.astype(int)])
    rows = []
    for name, model in {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
        "xgboost": XGBClassifier(n_estimators=100, max_depth=3, random_state=seed, n_jobs=1),
    }.items():
        model.fit(X_fit, y_fit)
        values = shap.TreeExplainer(model).shap_values(X_fit)
        if isinstance(values, list):  # one array per class
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:  # (rows, features, classes)
            values = values[:, :, 1]
        rows += [
            {"model": name, "feature": f, "mean_abs_shap": float(v)}
            for f, v in zip(X.columns, np.abs(values).mean(axis=0), strict=True)
        ]
    table = pd.DataFrame(rows).sort_values(["model", "mean_abs_shap"], ascending=[True, False])
    table.to_csv(out / "shap_importance.csv", index=False)
    rf = table[table["model"] == "random_forest"].iloc[::-1]
    plt.figure(figsize=(6, 4))
    plt.barh(rf["feature"], rf["mean_abs_shap"])
    plt.xlabel("Mean |SHAP|")
    plt.title("Random Forest feature importance (SHAP)")
    plt.savefig(out / "shap_random_forest.png", dpi=150, bbox_inches="tight")
    plt.close()
    return [
        "shap_importance.csv and shap_random_forest.png: "
        "models refitted on all data to explain them"
    ]


def _projections(X: pd.DataFrame, y: np.ndarray, seed: int, out: Path, private: bool) -> list[str]:
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    methods = {"PCA": PCA(n_components=2, random_state=seed)}
    if len(X) > 5:
        methods["t-SNE"] = TSNE(n_components=2, perplexity=min(10, len(X) - 1), random_state=seed)
    notes = []
    try:
        import umap

        methods["UMAP"] = umap.UMAP(n_neighbors=min(10, len(X) - 1), random_state=seed)
    except ImportError:
        notes.append("UMAP skipped: install the 'study' extra")
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4))
    for ax, (name, reducer) in zip(np.atleast_1d(axes), methods.items(), strict=True):
        coords = reducer.fit_transform(Z)
        for cls, colour in ((0, "tab:blue"), (1, "tab:red")):
            ax.scatter(*coords[y == cls].T, c=colour, s=25, label=["Normal", "PCOS"][cls])
        ax.set_title(name)
        ax.legend(fontsize=7)
    fig.savefig(out / "projections.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    privacy = " PRIVATE: one point per participant, do not publish" if private else ""
    return [f"projections.png: PCA/t-SNE/UMAP of real participants.{privacy}", *notes]


def write_analysis(out: Path, X: pd.DataFrame, y: np.ndarray, result: dict, cfg: dict) -> list[str]:
    seed = int(cfg["seed"])
    notes: list[str] = []
    mann_whitney(X, y).to_csv(out / "mann_whitney.csv", index=False)
    notes.append("mann_whitney.csv: real participants, PCOS vs normal, two-sided")
    notes += _curves(result["oof"], out)
    notes += _learning_curve(X, y, seed, out)
    notes += _shap(X, y, cfg, out)
    notes += _projections(X, y, seed, out, private=cfg["data_mode"] == "study")
    return notes
