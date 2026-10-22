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
from pcos_ecg.evaluation.cv import NotEstimableError, prepare_xy, run_experiment
from pcos_ecg.features.build import build_feature_table
from pcos_ecg.ingestion.load import (
    DataValidationError,
    load_covariates,
    load_labels,
    load_waveforms,
)
from pcos_ecg.ingestion.validate import validate_cohort, validate_waveforms
from pcos_ecg.reporting.report import write_report
from pcos_ecg.signal.preprocess import FilterConfigError

DEMO_DATA_CFG = "configs/demo_data.yaml"
DEMO_EXP_CFG = "configs/demo_experiment.yaml"


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
    table, qc = build_feature_table(waves, labels, data_cfg, load_covariates(data_cfg))

    out = resolve(data_cfg["output"]["features_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    resolve(data_cfg["output"]["qc_report_path"]).write_text(json.dumps(qc, indent=2))
    print(
        f"Features for {qc['n_records_quality_ok']}/{qc['n_records']} records "
        f"({qc['n_subjects']} participants) written to {out}"
    )
    return qc


def cmd_train(exp_cfg: dict, qc: dict | None = None) -> str:
    path = resolve(exp_cfg["features_path"])
    if not path.exists():
        raise DataValidationError(f"Feature table not found at {path}; run extract-features first")
    X, y, groups = prepare_xy(pd.read_csv(path), exp_cfg)
    with warnings.catch_warnings():
        # Undefined-metric warnings are expected on tiny folds; they are reported as NaN.
        warnings.simplefilter("ignore", category=UserWarning)
        result = run_experiment(X, y, groups, exp_cfg)
    out = write_report(result, exp_cfg, qc)
    print((out / "report.md").read_text())
    print(f"Report written to {out}")
    return str(out)


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
    d.add_argument("--subjects", type=int, default=40)
    d.add_argument("--positive", type=int, default=12)
    d.add_argument("--seed", type=int, default=7)

    for name, helptext in [
        ("validate-data", "check schema, sampling rates and labels (counts only)"),
        ("extract-features", "quality-check, filter and extract ECG features"),
        ("train-evaluate", "participant-grouped CV comparison of RF / SVM / XGBoost"),
    ]:
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--config", required=True, help="path to a YAML config")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo":
            cmd_demo(args.subjects, args.positive, args.seed)
        elif args.command == "validate-data":
            cmd_validate(load_config(args.config))
        elif args.command == "extract-features":
            cmd_extract(load_config(args.config))
        elif args.command == "train-evaluate":
            cmd_train(load_config(args.config))
    except (ConfigError, DataValidationError, FilterConfigError, NotEstimableError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
