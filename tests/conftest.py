"""Shared fixtures. Every fixture is synthetic; no test touches real participant data."""

from pathlib import Path

import pytest
import yaml

from pcos_ecg.demo.synthetic import make_demo_cohort, make_demo_covariates

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def small_cohort():
    return make_demo_cohort(
        n_subjects=16, n_positive=6, records_per_subject=2, duration_s=20, seed=3
    )


@pytest.fixture
def data_cfg(tmp_path, small_cohort):
    waves, labels = small_cohort
    waves.to_csv(tmp_path / "ecg_long.csv", index=False)
    labels.to_csv(tmp_path / "labels.csv", index=False)
    make_demo_covariates(labels, seed=3).to_csv(tmp_path / "covariates.csv", index=False)
    cfg = yaml.safe_load((REPO / "configs" / "demo_data.yaml").read_text())
    cfg["waveforms"]["path"] = str(tmp_path / "ecg_long.csv")
    cfg["labels"]["path"] = str(tmp_path / "labels.csv")
    cfg["covariates"]["path"] = str(tmp_path / "covariates.csv")
    # The fast built-in backend keeps the core tests independent of the 'study' extra.
    cfg["signal"] = {"backend": "scipy"}
    cfg["output"] = {
        "features_path": str(tmp_path / "features.csv"),
        "qc_report_path": str(tmp_path / "qc.json"),
    }
    return cfg


@pytest.fixture
def exp_cfg(tmp_path):
    cfg = yaml.safe_load((REPO / "configs" / "demo_experiment.yaml").read_text())
    cfg["features_path"] = str(tmp_path / "features.csv")
    cfg["output_dir"] = str(tmp_path / "runs")
    cfg["cv"]["max_splits"] = 3
    # A lighter run of the same pipeline: fewer conditions and models, no CTGAN or Boruta.
    cfg["conditions"] = ["study_synthesis", "real_only"]
    cfg["models"] = ["random_forest", "baseline"]
    cfg["feature_selection"] = {"method": "mutual_info", "fallback_k": 3}
    cfg["synthesis"]["compare_methods"] = ["class_conditional_gaussian"]
    return cfg
