import pandas as pd
import pytest

from pcos_ecg.ingestion.load import DataValidationError, load_labels, load_waveforms
from pcos_ecg.ingestion.validate import validate_cohort, validate_waveforms


def test_valid_fixture_loads(data_cfg):
    waves = load_waveforms(data_cfg)
    report = validate_waveforms(waves)
    assert report["n_subjects"] == 16
    assert report["sample_rates_hz"] == [250.0]
    cohort = validate_cohort(load_labels(data_cfg), waves["subject_id"])
    assert (cohort["n_positive"], cohort["n_negative"]) == (6, 10)


def test_missing_column_is_actionable(data_cfg):
    data_cfg["waveforms"]["columns"]["amplitude"] = "voltage"
    with pytest.raises(DataValidationError, match="voltage"):
        load_waveforms(data_cfg)


def test_missing_file_points_to_docs(data_cfg):
    data_cfg["waveforms"]["path"] = "/nonexistent/ecg.csv"
    with pytest.raises(DataValidationError, match="data/private"):
        load_waveforms(data_cfg)


def test_microvolts_converted_to_millivolts(data_cfg, tmp_path):
    waves = pd.read_csv(data_cfg["waveforms"]["path"])
    waves["amplitude"] *= 1000
    waves.to_csv(tmp_path / "uv.csv", index=False)
    data_cfg["waveforms"].update(path=str(tmp_path / "uv.csv"), amplitude_unit="uV")
    mv = load_waveforms(data_cfg)["amplitude"]
    assert mv.abs().max() < 5


def test_implausible_sampling_rate_rejected(data_cfg):
    waves = load_waveforms(data_cfg)
    waves["sample_rate_hz"] = 5.0
    with pytest.raises(DataValidationError, match="sampling rate"):
        validate_waveforms(waves)


def test_record_shared_by_two_participants_rejected(data_cfg):
    waves = load_waveforms(data_cfg)
    first = waves["record_id"].iloc[0]
    waves.loc[waves.index[:10], "subject_id"] = "OTHER"
    assert waves.loc[waves["record_id"] == first, "subject_id"].nunique() == 2
    with pytest.raises(DataValidationError, match="more than|>1"):
        validate_waveforms(waves)


def test_conflicting_labels_rejected(data_cfg, tmp_path):
    labels = pd.read_csv(data_cfg["labels"]["path"])
    dup = labels.iloc[[0]].assign(label=1 - labels["label"].iloc[0])
    pd.concat([labels, dup]).to_csv(tmp_path / "bad.csv", index=False)
    data_cfg["labels"]["path"] = str(tmp_path / "bad.csv")
    with pytest.raises(DataValidationError, match="conflicting"):
        load_labels(data_cfg)


def test_non_binary_label_rejected(data_cfg, tmp_path):
    labels = pd.read_csv(data_cfg["labels"]["path"])
    labels.loc[0, "label"] = 2
    labels.to_csv(tmp_path / "bad.csv", index=False)
    data_cfg["labels"]["path"] = str(tmp_path / "bad.csv")
    with pytest.raises(DataValidationError, match="binary"):
        load_labels(data_cfg)


def test_validation_messages_do_not_leak_ids(data_cfg):
    waves = load_waveforms(data_cfg)
    labels = load_labels(data_cfg).iloc[2:]  # two participants unlabelled
    report = validate_cohort(labels, waves["subject_id"])
    assert "SYN" not in str(report)
