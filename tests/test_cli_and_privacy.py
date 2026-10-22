import json
import subprocess
from pathlib import Path

import pytest

from pcos_ecg.cli import cmd_extract, cmd_train, main

REPO = Path(__file__).resolve().parents[1]


def test_help_runs(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "not a diagnostic tool" in capsys.readouterr().out


def test_missing_study_file_errors_without_demo_fallback(tmp_path, capsys):
    cfg = tmp_path / "local.yaml"
    cfg.write_text(
        (REPO / "configs" / "experiment.example.yaml")
        .read_text()
        .replace("data/private/features.csv", str(tmp_path / "absent.csv"))
    )
    assert main(["train-evaluate", "--config", str(cfg)]) == 2
    assert "extract-features" in capsys.readouterr().err


def test_end_to_end_reports_contain_no_ids_or_rows(data_cfg, exp_cfg):
    qc = cmd_extract(data_cfg)
    out = Path(cmd_train(exp_cfg, qc))
    for name in ("metadata.json", "per_fold.csv", "summary.csv", "report.md"):
        text = (out / name).read_text()
        assert "SYN0" not in text
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["data_mode"] == "synthetic_demo"
    assert "SYNTHETIC DEMO" in meta["caveat"]
    assert "SYN0" not in Path(data_cfg["output"]["qc_report_path"]).read_text()


def test_private_paths_are_git_ignored():
    ignore = (REPO / ".gitignore").read_text()
    for pattern in ("data/private/", "data/demo/", "runs/", ".env", "configs/local_*.yaml"):
        assert pattern in ignore


def test_no_data_files_are_tracked():
    try:
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    risky = [
        f
        for f in tracked
        if f.startswith(("data/private", "runs/"))
        or f.endswith((".csv", ".edf", ".dat", ".joblib", ".pkl"))
    ]
    assert risky == []
