"""Write experiment outputs to runs/<timestamp>_<data_mode>/.

Reports contain aggregate counts and metrics only: no participant IDs, no feature rows,
no signal samples.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from pcos_ecg.config import resolve

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost")

DEMO_CAVEAT = (
    "SYNTHETIC DEMO: these numbers describe a toy generator with arbitrary class "
    "differences. They are not evidence about PCOS or about the RIT study."
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


def write_report(result: dict, cfg: dict, qc: dict | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = resolve(cfg.get("output_dir", "runs")) / f"{stamp}_{cfg['data_mode']}"
    out.mkdir(parents=True, exist_ok=False)

    caveat = DEMO_CAVEAT if cfg["data_mode"] == "synthetic_demo" else STUDY_CAVEAT
    meta = {
        "created_utc": stamp,
        "data_mode": cfg["data_mode"],
        "caveat": caveat,
        "seed": cfg["seed"],
        "split_strategy": result["split_strategy"],
        "n_splits": result["n_splits"],
        "n_participants": result["n_participants"],
        "n_positive_participants": result["n_positive_participants"],
        "decision_threshold": 0.5,
        "positive_label": "1 = PCOS (study) / synthetic positive class (demo)",
        "models": cfg["models"],
        "conditions": cfg["conditions"],
        "feature_selection": cfg.get("feature_selection"),
        "feature_selection_counts": result["feature_selection_counts"],
        "warnings": result["warnings"],
        "versions": _versions(),
        "qc": qc,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    result["per_fold"].to_csv(out / "per_fold.csv", index=False)
    result["summary"].to_csv(out / "summary.csv", index=False)
    (out / "report.md").write_text(_markdown(meta, result["summary"]))
    return out


def _markdown(meta: dict, summary: pd.DataFrame) -> str:
    cols = ["model", "condition"] + [
        c for c in summary.columns if c.endswith("_mean") or c.endswith("_std")
    ]
    lines = [
        f"# Model comparison ({meta['data_mode']})",
        "",
        f"> {meta['caveat']}",
        "",
        f"- Created (UTC): {meta['created_utc']}",
        f"- Participants: {meta['n_participants']} ({meta['n_positive_participants']} positive)",
        f"- Evaluation: {meta['split_strategy']}, {meta['n_splits']} folds, seed {meta['seed']}",
        "- Threshold: 0.5, fixed in advance (not tuned)",
        "- Metrics: mean and SD across folds; NaN = undefined for that fold",
        "",
        _table(summary[cols]),
        "",
    ]
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
