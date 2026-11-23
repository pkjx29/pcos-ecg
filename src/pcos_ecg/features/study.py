"""The original study's participant-level feature table.

Column names follow the study's notebooks (original_study/03_Code) so results can be compared
directly. One row per participant: ECG features are the median over that participant's
quality-checked recordings; physiological features come from a separate measurement or
questionnaire file and are never derived from the ECG.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Measured from the ECG. Units follow the study: SDNN in ms, everything else in seconds.
ECG_MEASURED = {
    "HRV_SDNN_Estimate": "ms",
    "Mean_RR_Interval": "s",
    "QT_Interval": "s",
    "QTc_Bazett": "s",
    "PR_Interval": "s",
    "QRS_Duration": "s",
    "P_Wave_Dispersion": "s",
    "T_Wave_Duration": "s",
}

# Engineered from the measured ECG features (report section 3.5.4).
ENGINEERED = {
    "SDNN_MeanRR_Ratio": "unitless",
    "QTc_MeanRR_Ratio": "1/s",
    "PWaveDisp_x_QTc": "s^2",
    "HRVSDNN_x_QTc": "ms*s",
}

PHYSIOLOGICAL = {
    "Cycle_Length": "days",
    "Cycle_Regularity": "0 = regular, 1 = irregular",
    "BMI": "kg/m^2",
    "Family_History": "0 = no, 1 = yes (PCOS or diabetes)",
}

# The final 10 features the study kept after feature selection (report Table 3.5.5b).
FINAL_ECG = [
    "HRV_SDNN_Estimate",
    "P_Wave_Dispersion",
    "SDNN_MeanRR_Ratio",
    "QTc_Bazett",
    "PR_Interval",
    "T_Wave_Duration",
]
FINAL_PHYSIOLOGICAL = ["Cycle_Length", "Cycle_Regularity", "BMI", "Family_History"]

BINARY_CODES = {
    "Cycle_Regularity": {"regular": 0, "irregular": 1},
    "Family_History": {"no": 0, "yes": 1},
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise formulas only, so this is safe to apply before any train/test split.

    SDNN is stored in ms but RR in s, so the ratio converts SDNN to seconds first; the result
    is the coefficient of variation of RR. The report does not state the units it used.
    """
    out = df.copy()
    sdnn_s = out["HRV_SDNN_Estimate"] / 1000.0
    rr = out["Mean_RR_Interval"].where(out["Mean_RR_Interval"] > 0)
    out["SDNN_MeanRR_Ratio"] = sdnn_s / rr
    out["QTc_MeanRR_Ratio"] = out["QTc_Bazett"] / rr
    out["PWaveDisp_x_QTc"] = out["P_Wave_Dispersion"] * out["QTc_Bazett"]
    out["HRVSDNN_x_QTc"] = out["HRV_SDNN_Estimate"] * out["QTc_Bazett"]
    return out


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Map text answers (Regular/Irregular, Yes/No) to 0/1; numbers are left as they are."""
    out = df.copy()
    for col, codes in BINARY_CODES.items():
        if col in out and not pd.api.types.is_numeric_dtype(out[col]):
            text = out[col].astype("string").str.strip().str.lower()
            unknown = set(text.dropna()) - set(codes)
            if unknown:
                raise ValueError(f"{col} has unexpected values: {sorted(unknown)}")
            out[col] = text.map(codes).astype(float)
    return out


def participant_table(
    records: pd.DataFrame,
    labels: pd.DataFrame,
    covariates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Collapse record-level ECG features to one row per participant and add the rest."""
    ok = records[records["quality_ok"]] if "quality_ok" in records else records
    ecg_cols = [c for c in ECG_MEASURED if c in ok]
    table = ok.groupby("subject_id")[ecg_cols].median().reset_index()
    for col in ECG_MEASURED:
        if col not in table:
            table[col] = np.nan
    table = add_engineered_features(table).merge(labels, on="subject_id", how="inner")
    if covariates is not None:
        table = table.merge(encode_binary(covariates), on="subject_id", how="left")
    return table


def feature_sets(cfg: dict | None = None) -> dict[str, list[str]]:
    """ECG-only, physiology-only and combined sets; the config may override them."""
    sets = (cfg or {}).get("feature_sets")
    if sets:
        return {name: list(cols) for name, cols in sets.items()}
    return {
        "ECG Only": FINAL_ECG,
        "Physiology Only": FINAL_PHYSIOLOGICAL,
        "Combined": FINAL_ECG + FINAL_PHYSIOLOGICAL,
    }
