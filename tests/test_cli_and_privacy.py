import csv
import json
import re
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
    text_files = [p for p in out.iterdir() if p.suffix in {".csv", ".json", ".md"}]
    assert {"metadata.json", "summary.csv", "report.md", "folds.csv"} <= {
        p.name for p in text_files
    }
    for path in text_files:
        assert "SYN0" not in path.read_text(), path.name
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["data_mode"] == "synthetic_demo"
    assert "SYNTHETIC DEMO" in meta["caveat"]
    assert "SYN0" not in Path(data_cfg["output"]["qc_report_path"]).read_text()
    assert (out / "ensemble_demo_only.joblib").exists()


def test_ensemble_is_never_saved_for_study_data(data_cfg, exp_cfg):
    cmd_extract(data_cfg)
    exp_cfg["data_mode"] = "study"
    exp_cfg["analysis"] = {"enabled": False}
    out = Path(cmd_train(exp_cfg))
    assert not list(out.glob("*.joblib"))
    assert "SYNTHETIC DEMO" not in (out / "report.md").read_text()


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
        or (f.endswith((".csv", ".edf", ".dat", ".joblib", ".pkl")) and not _is_aggregate_result(f))
    ]
    assert risky == []


AGGREGATE_DIR = "original_study/04_Results/"
ID_COLUMN = re.compile(r"subject|participant|record_id|patient|\bname\b", re.IGNORECASE)
# The original feature selection wrongly fed Subject_ID in as a feature, so this matrix has a
# "cat__Subject_ID" column. It holds correlation coefficients, not identifiers.
REVIEWED_TABLES = {AGGREGATE_DIR + "04_Feature_Selection_Results/tables/correlation_matrix.csv"}


def _is_aggregate_result(path: str) -> bool:
    """Published summary tables from the original study: CSVs whose columns carry no IDs."""
    if not (path.startswith(AGGREGATE_DIR) and path.endswith(".csv")):
        return False
    if path in REVIEWED_TABLES:
        return True
    with (REPO / path).open(encoding="utf-8", errors="replace") as fh:
        header = next(csv.reader(fh), [])
    return not any(ID_COLUMN.search(col) for col in header)


def test_original_study_has_no_personal_details():
    root = REPO / "original_study"
    if not root.exists():
        pytest.skip("original_study/ not present")
    pattern = re.compile(
        r"[\w.+-]+@[\w-]+\.(com|in|org|edu)|/content/drive|drive\.mount|/Users/", re.IGNORECASE
    )
    hits = [
        f"{p.relative_to(REPO)}: {m.group(0)}"
        for p in root.rglob("*")
        if p.suffix in {".py", ".ipynb", ".csv", ".txt", ".json", ".md"}
        for m in [pattern.search(p.read_text(encoding="utf-8", errors="replace"))]
        if m
    ]
    assert hits == []
