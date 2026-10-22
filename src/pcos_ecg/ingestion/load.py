"""Load waveform, label and covariate files into canonical column names.

Canonical waveform columns: subject_id, record_id, lead, sample_index, sample_rate_hz,
amplitude (mV). Canonical label columns: subject_id, label (1 = positive, 0 = negative).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pcos_ecg.config import ConfigError, resolve

WAVEFORM_COLUMNS = [
    "subject_id",
    "record_id",
    "lead",
    "sample_index",
    "sample_rate_hz",
    "amplitude",
]
UNIT_TO_MV = {"mV": 1.0, "uV": 1e-3}


class DataValidationError(ValueError):
    """Raised when input data cannot be used. Messages never contain sample values."""


def _read_csv(path_value: Any, what: str) -> pd.DataFrame:
    path = resolve(path_value)
    if path is None:
        raise ConfigError(f"No path configured for {what}")
    if not path.exists():
        raise DataValidationError(
            f"{what} file not found at {path}. Study data belongs in data/private/ "
            "(see data/README.md); run `pcos-ecg demo` for the synthetic demo."
        )
    return pd.read_csv(path)


def load_waveforms(cfg: dict) -> pd.DataFrame:
    wcfg = cfg["waveforms"]
    mapping: dict[str, str] = wcfg["columns"]
    missing_keys = set(WAVEFORM_COLUMNS) - set(mapping)
    if missing_keys:
        raise ConfigError(f"waveforms.columns is missing mappings for {sorted(missing_keys)}")

    df = _read_csv(wcfg["path"], "Waveform")
    absent = [src for src in mapping.values() if src not in df.columns]
    if absent:
        raise DataValidationError(f"Waveform file lacks configured columns: {absent}")
    df = df[list(mapping.values())].rename(columns={v: k for k, v in mapping.items()})

    unit = wcfg.get("amplitude_unit", "mV")
    if unit not in UNIT_TO_MV:
        raise ConfigError(f"amplitude_unit must be one of {list(UNIT_TO_MV)}, got {unit!r}")
    df["amplitude"] = pd.to_numeric(df["amplitude"], errors="coerce") * UNIT_TO_MV[unit]

    lead = wcfg.get("lead")
    if lead is not None:
        df["lead"] = df["lead"].astype(str)
        if lead not in set(df["lead"]):
            raise DataValidationError(f"Lead {lead!r} not present in waveform file")
        df = df[df["lead"] == lead]
    df["subject_id"] = df["subject_id"].astype(str)
    df["record_id"] = df["record_id"].astype(str)
    return df.reset_index(drop=True)


def load_labels(cfg: dict) -> pd.DataFrame:
    lcfg = cfg["labels"]
    df = _read_csv(lcfg["path"], "Label")
    sid, lab = lcfg["subject_id"], lcfg["label"]
    absent = [c for c in (sid, lab) if c not in df.columns]
    if absent:
        raise DataValidationError(f"Label file lacks configured columns: {absent}")
    df = df[[sid, lab]].rename(columns={sid: "subject_id", lab: "raw_label"})
    df["subject_id"] = df["subject_id"].astype(str)

    if df["raw_label"].isna().any():
        raise DataValidationError(
            f"{int(df['raw_label'].isna().sum())} participant(s) lack a label"
        )
    values = set(df["raw_label"].unique())
    if len(values) > 2:
        raise DataValidationError(f"Expected a binary label, found {len(values)} distinct values")
    pos = lcfg.get("positive_value", 1)
    if pos not in values:
        raise DataValidationError(f"positive_value {pos!r} does not occur in the label column")

    conflicting = df.groupby("subject_id")["raw_label"].nunique()
    if (conflicting > 1).any():
        raise DataValidationError(
            f"{int((conflicting > 1).sum())} participant(s) have conflicting labels"
        )
    df = df.drop_duplicates("subject_id")
    df["label"] = (df["raw_label"] == pos).astype(int)
    return df[["subject_id", "label"]].reset_index(drop=True)


def load_covariates(cfg: dict) -> pd.DataFrame | None:
    ccfg = cfg.get("covariates") or {}
    if not ccfg.get("path"):
        return None
    df = _read_csv(ccfg["path"], "Covariate")
    sid, cols = ccfg["subject_id"], list(ccfg.get("columns") or [])
    absent = [c for c in [sid, *cols] if c not in df.columns]
    if absent:
        raise DataValidationError(f"Covariate file lacks configured columns: {absent}")
    df = df[[sid, *cols]].rename(columns={sid: "subject_id"})
    df["subject_id"] = df["subject_id"].astype(str)
    if df["subject_id"].duplicated().any():
        raise DataValidationError("Covariate file has more than one row per participant")
    return df
