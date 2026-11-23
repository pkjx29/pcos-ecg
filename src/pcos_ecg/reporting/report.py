"""Write experiment outputs to runs/<timestamp>_<data_mode>/.

Reports contain aggregate counts and metrics only: no participant IDs, no feature rows,
no per-person predictions, no signal samples. Plots that show one point per participant
(PCA/t-SNE/UMAP) are written to the private runs/ folder and flagged in the report.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from pcos_ecg.config import resolve
from pcos_ecg.models.factory import CONDITION_LABELS, MODEL_LABELS

PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "imbalanced-learn",
    "xgboost",
    "sdv",
    "sdmetrics",
    "boruta",
    "neurokit2",
    "shap",
    "umap-learn",
)

DEMO_CAVEAT = (
    "SYNTHETIC DEMO: these numbers describe a toy generator with arbitrary class "
    "differences. They say nothing about PCOS or about the RIT study."
)
STUDY_CAVEAT = (
    "Internal cross-validation on a small cohort: estimates are unstable and are not an "
    "external or clinical validation. Research use only; not a diagnostic tool."
)


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def new_run_dir(cfg: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = resolve(cfg.get("output_dir", "runs")) / f"{stamp}_{cfg['data_mode']}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def write_report(
    out: Path,
    result: dict,
    cfg: dict,
    qc: dict | None = None,
    synth_comparison: pd.DataFrame | None = None,
    analysis_notes: list[str] | None = None,
) -> Path:
    caveat = DEMO_CAVEAT if cfg["data_mode"] == "synthetic_demo" else STUDY_CAVEAT
    meta = {
        "created_utc": out.name.split("_")[0],
        "data_mode": cfg["data_mode"],
        "caveat": caveat,
        "seed": cfg["seed"],
        "split_strategy": result["strategy"],
        "n_splits": result["n_splits"],
        "n_participants": result["n_participants"],
        "n_positive_participants": result["n_positive_participants"],
        "decision_threshold": 0.5,
        "positive_label": "1 = PCOS (study) / synthetic positive class (demo)",
        "models": cfg.get("models"),
        "conditions": cfg.get("conditions"),
        "synthesis": cfg.get("synthesis"),
        "feature_selection": cfg.get("feature_selection"),
        "feature_selection_fallbacks": result["fallback_count"],
        "warnings": result["warnings"],
        "analysis_notes": analysis_notes or [],
        "versions": _versions(),
        "qc": qc,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    result["summary"].to_csv(out / "summary.csv", index=False)
    result["folds"].to_csv(out / "folds.csv", index=False)
    result["selection_frequency"].to_csv(out / "feature_selection_frequency.csv", index=False)
    if synth_comparison is not None:
        synth_comparison.to_csv(out / "synthesizer_comparison.csv", index=False)
    (out / "report.md").write_text(_markdown(meta, result["summary"], synth_comparison))
    return out


def _markdown(meta: dict, summary: pd.DataFrame, synth: pd.DataFrame | None) -> str:
    show = summary.assign(
        model=summary["model"].map(MODEL_LABELS).fillna(summary["model"]),
    )
    cols = [
        "model",
        "feature_set",
        "roc_auc",
        "pr_auc",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
    ]
    lines = [
        f"# Model comparison ({meta['data_mode']})",
        "",
        f"> {meta['caveat']}",
        "",
        f"- Created (UTC): {meta['created_utc']}",
        f"- Participants: {meta['n_participants']} ({meta['n_positive_participants']} positive); "
        "every test row is a real, held-out participant",
        f"- Evaluation: {meta['split_strategy']} ({meta['n_splits']} folds), seed {meta['seed']}; "
        "metrics on pooled out-of-fold predictions",
        "- Threshold: 0.5, fixed in advance (not tuned)",
        "- Feature selection inside each fold: "
        f"{(meta['feature_selection'] or {}).get('method', 'study')}"
        f" (fell back to mutual information {meta['feature_selection_fallbacks']} time(s))",
        "",
    ]
    for condition, part in show.groupby("condition", sort=False):
        lines += [f"## {CONDITION_LABELS.get(condition, condition)}", "", _table(part[cols]), ""]
    if synth is not None:
        lines += [
            "## Synthesizer comparison (as in the study's 03_1)",
            "",
            _table(synth.round(4)),
            "",
        ]
    if meta["analysis_notes"]:
        lines += ["## Analysis outputs", "", *[f"- {n}" for n in meta["analysis_notes"]], ""]
    if meta["warnings"]:
        lines += ["## Warnings", "", *[f"- {w}" for w in meta["warnings"]], ""]
    return "\n".join(lines)


def _table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def latest_run(output_dir: str | Path = "runs") -> Path | None:
    root = resolve(output_dir)
    if not root.exists():
        return None
    runs = sorted(p for p in root.iterdir() if (p / "metadata.json").exists())
    return runs[-1] if runs else None
