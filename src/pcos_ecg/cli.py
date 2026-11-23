"""Command-line entry point: `pcos-ecg <command>` or `python -m pcos_ecg <command>`.

Study-data commands never fall back to demo data: a missing file is an error.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import pandas as pd

from pcos_ecg.config import ConfigError, load_config, resolve
from pcos_ecg.evaluation.cv import NotEstimableError, prepare_table, run_experiment
from pcos_ecg.features.build import build_feature_table
from pcos_ecg.features.study import BINARY_CODES, participant_table
from pcos_ecg.ingestion.load import (
    DataValidationError,
    load_covariates,
    load_labels,
    load_waveforms,
)
from pcos_ecg.ingestion.validate import validate_cohort, validate_waveforms
from pcos_ecg.reporting.report import new_run_dir, write_report
from pcos_ecg.signal.preprocess import FilterConfigError

DEMO_DATA_CFG = "configs/demo_data.yaml"
DEMO_EXP_CFG = "configs/demo_experiment.yaml"


def cmd_import_kardia(mapping: str, out: str) -> dict:
    from pcos_ecg.ingestion.kardia_pdf import import_kardia_folder

    summary = import_kardia_folder(resolve(mapping), resolve(out))
    print(json.dumps(summary, indent=2))
    return summary


def cmd_validate(data_cfg: dict) -> dict:
    waves = load_waveforms(data_cfg)
    labels = load_labels(data_cfg)
    report = {
        "data_mode": data_cfg["data_mode"],
        "waveforms": validate_waveforms(waves),
        "cohort": validate_cohort(labels, waves["subject_id"]),
    }
    print(json.dumps(report, indent=2))
    return report


def cmd_extract(data_cfg: dict) -> dict:
    waves = load_waveforms(data_cfg)
    labels = load_labels(data_cfg)
    validate_waveforms(waves)
    validate_cohort(labels, waves["subject_id"])
    records, qc = build_feature_table(waves, data_cfg)
    table = participant_table(records, labels, load_covariates(data_cfg))

    out = resolve(data_cfg["output"]["features_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    resolve(data_cfg["output"]["qc_report_path"]).write_text(json.dumps(qc, indent=2))
    print(
        f"Features for {qc['n_records_quality_ok']}/{qc['n_records']} records "
        f"({len(table)} participants, {qc['signal_backend']} backend) written to {out}"
    )
    return qc


def _load_table(exp_cfg: dict):
    path = resolve(exp_cfg["features_path"])
    if not path.exists():
        raise DataValidationError(f"Feature table not found at {path}; run extract-features first")
    return prepare_table(pd.read_csv(path), exp_cfg)


def _compare(X, y, exp_cfg: dict) -> pd.DataFrame | None:
    s = exp_cfg.get("synthesis") or {}
    methods = s.get("compare_methods")
    if not methods:
        return None
    from pcos_ecg.synthesis.compare import compare_synthesizers

    return compare_synthesizers(
        X,
        y,
        methods,
        n_rows=s.get("compare_rows", 150),
        categorical=[c for c in BINARY_CODES if c in X.columns],
        seed=int(exp_cfg["seed"]),
        epochs=s.get("ctgan_epochs", 300),
    )


def cmd_compare(exp_cfg: dict) -> str:
    X, y, _ = _load_table(exp_cfg)
    table = _compare(X, y, exp_cfg)
    if table is None:
        raise ConfigError("set synthesis.compare_methods in the experiment config")
    out = new_run_dir(exp_cfg)
    table.to_csv(out / "synthesizer_comparison.csv", index=False)
    print(table.round(4).to_string(index=False))
    print(f"Written to {out}")
    return str(out)


def cmd_train(exp_cfg: dict, qc: dict | None = None) -> str:
    X, y, groups = _load_table(exp_cfg)
    with warnings.catch_warnings():
        # Undefined-metric warnings are expected on small folds; they are reported as NaN.
        warnings.simplefilter("ignore", category=UserWarning)
        result = run_experiment(X, y, groups, exp_cfg)
        synth = _compare(X, y, exp_cfg)
    out = new_run_dir(exp_cfg)
    notes = []
    if (exp_cfg.get("analysis") or {}).get("enabled", True):
        from pcos_ecg.analysis.outputs import write_analysis

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            notes = write_analysis(out, X, y, result, exp_cfg)
    if exp_cfg["data_mode"] == "synthetic_demo" and (exp_cfg.get("ensemble") or {}).get("enabled"):
        notes.append(_save_demo_ensemble(out, X, y, exp_cfg))
    write_report(out, result, exp_cfg, qc, synth, notes)
    print((out / "report.md").read_text())
    print(f"Report written to {out}")
    return str(out)


def _save_demo_ensemble(out, X, y, exp_cfg: dict) -> str:
    """Fit the 0.7/0.3 ensemble on all demo data and save it. Never done for study data."""
    import joblib

    from pcos_ecg.features.selection import select_features
    from pcos_ecg.features.study import feature_sets
    from pcos_ecg.models.ensemble import WeightedSoftVote

    sets, fs = feature_sets(exp_cfg), exp_cfg.get("feature_selection") or {}
    pick = {
        name: select_features(
            X[sets[name]], y, method=fs.get("method", "study"), seed=int(exp_cfg["seed"])
        ).selected
        for name in ("ECG Only", "Physiology Only")
    }
    weights = tuple((exp_cfg.get("ensemble") or {}).get("weights", (0.7, 0.3)))
    model = WeightedSoftVote(
        pick["ECG Only"], pick["Physiology Only"], weights, int(exp_cfg["seed"])
    )
    joblib.dump(model.fit(X, y), out / "ensemble_demo_only.joblib")
    return "ensemble_demo_only.joblib: fitted on synthetic demo data only; not a PCOS model"


def cmd_demo(n_subjects: int, n_positive: int, seed: int) -> str:
    from pcos_ecg.demo.synthetic import write_demo_cohort

    data_cfg, exp_cfg = load_config(DEMO_DATA_CFG), load_config(DEMO_EXP_CFG)
    for cfg in (data_cfg, exp_cfg):
        if cfg["data_mode"] != "synthetic_demo":
            raise ConfigError("demo configs must have data_mode: synthetic_demo")
    out_dir = resolve(data_cfg["waveforms"]["path"]).parent
    write_demo_cohort(out_dir, n_subjects=n_subjects, n_positive=n_positive, seed=seed)
    print(f"Synthetic demo cohort written to {out_dir} (no real participants)")
    qc = cmd_extract(data_cfg)
    return cmd_train(exp_cfg, qc)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pcos-ecg",
        description="ECG feature pipeline and model comparison for PCOS screening research. "
        "Research prototype; not a diagnostic tool.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="generate synthetic data and run the whole pipeline")
    d.add_argument("--subjects", type=int, default=30)
    d.add_argument("--positive", type=int, default=10)
    d.add_argument("--seed", type=int, default=7)

    k = sub.add_parser(
        "import-kardia", help="convert Kardia PDF exports into the long-form waveform file"
    )
    k.add_argument("--mapping", required=True, help="CSV with columns file, subject_id, record_id")
    k.add_argument("--out", required=True, help="output CSV, e.g. data/private/ecg_long.csv")

    for name, helptext in [
        ("validate-data", "check schema, sampling rates and labels (counts only)"),
        ("extract-features", "quality-check recordings and build the participant feature table"),
        ("compare-synthesizers", "score the study's synthesizers as in its 03_1 notebook"),
        (
            "train-evaluate",
            "leakage-safe comparison of RF / XGBoost / SVM on held-out participants",
        ),
    ]:
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--config", required=True, help="path to a YAML config")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo":
            cmd_demo(args.subjects, args.positive, args.seed)
        elif args.command == "import-kardia":
            cmd_import_kardia(args.mapping, args.out)
        elif args.command == "validate-data":
            cmd_validate(load_config(args.config))
        elif args.command == "extract-features":
            cmd_extract(load_config(args.config))
        elif args.command == "compare-synthesizers":
            cmd_compare(load_config(args.config))
        elif args.command == "train-evaluate":
            cmd_train(load_config(args.config))
    except (ConfigError, DataValidationError, FilterConfigError, NotEstimableError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:  # KardiaPDFError and malformed inputs
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
