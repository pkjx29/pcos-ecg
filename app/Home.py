"""Streamlit research dashboard. Run from the repo root: `streamlit run app/Home.py`.

Demo-safe by default: it shows ECG processing for one recording at a time and model
metrics only from report files produced by `pcos-ecg`. It never outputs a PCOS
prediction for an individual.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from pcos_ecg.demo.synthetic import SubjectParams, synth_ecg  # noqa: E402
from pcos_ecg.reporting.report import latest_run  # noqa: E402
from pcos_ecg.signal.features import FEATURE_UNITS, extract_features  # noqa: E402
from pcos_ecg.signal.peaks import detect_r_peaks  # noqa: E402
from pcos_ecg.signal.preprocess import (  # noqa: E402
    FilterConfigError,
    bandpass_notch,
    check_quality,
)

NOTICE = (
    "Research prototype. Not a medical device and not a diagnostic tool. "
    "Nothing here should inform a health decision."
)

st.set_page_config(page_title="PCOS-ECG research pipeline", layout="wide")
st.title("Early screening of PCOS from ECG waveforms")
st.warning(NOTICE)

tab_over, tab_ecg, tab_models, tab_limits = st.tabs(
    ["Overview", "ECG quality and features", "Model comparison", "Data and limitations"]
)

with tab_over:
    st.markdown(
        """
This dashboard accompanies a student research project at RIT Bengaluru that compared
Random Forest, SVM and XGBoost on ECG-derived features for PCOS screening research.

**Pipeline:** load and validate data → quality checks → zero-phase band-pass and 50 Hz
notch → R-peak detection → rhythm and interval features → participant-grouped stratified
cross-validation, with feature selection and SMOTE fitted inside each training fold.

The public repository contains **no volunteer data**. Everything shown here comes from a
synthetic generator unless you run the pipeline locally on approved study data.
"""
    )

with tab_ecg:
    st.subheader("Check one recording")
    st.caption(
        "Uploads are processed in memory and are not saved. Use only data you are allowed "
        "to process; do not upload identifiable recordings to a public deployment."
    )
    source = st.radio("Source", ["Synthetic example", "Upload CSV"], horizontal=True)
    x, fs = None, None
    if source == "Synthetic example":
        c1, c2 = st.columns(2)
        hr = c1.slider("Heart rate (bpm)", 50, 110, 72)
        seed = c2.number_input("Seed", 0, 10_000, 0)
        x, _ = synth_ecg(
            SubjectParams(hr, 0.04, 0.41, 0.16, 1.1), 30, 250.0, np.random.default_rng(seed)
        )
        fs = 250.0
        st.info("Synthetic ECG-like signal; not a real recording.")
    else:
        up = st.file_uploader("CSV with one amplitude column in mV", type=["csv"])
        fs = st.number_input("Sampling rate (Hz)", 50.0, 10_000.0, 250.0)
        if up is not None:
            df = pd.read_csv(io.BytesIO(up.getvalue()))
            col = st.selectbox("Amplitude column (mV)", df.columns)
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)

    if x is not None:
        quality = check_quality(x, fs)
        if not quality.ok:
            st.error("Recording rejected: " + "; ".join(quality.reasons))
        else:
            try:
                y = bandpass_notch(x, fs)
            except FilterConfigError as exc:
                st.error(str(exc))
                st.stop()
            peaks = detect_r_peaks(y, fs)
            t = np.arange(y.size) / fs
            fig, ax = plt.subplots(figsize=(11, 3))
            view = t < 10
            ax.plot(t[view], x[view], color="0.75", lw=0.8, label="raw")
            ax.plot(t[view], y[view], lw=1.0, label="filtered (zero-phase, offline)")
            pk = peaks[peaks < view.sum()]
            ax.plot(t[pk], y[pk], "v", color="crimson", label="detected R peaks (estimate)")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("amplitude (mV)")
            ax.set_title(f"First 10 s, fs = {fs:g} Hz")
            ax.legend(loc="upper right", fontsize=8)
            st.pyplot(fig)

            res = extract_features(y, peaks, fs)
            rows = [
                {
                    "feature": k,
                    "value": None if pd.isna(v) else round(float(v), 4),
                    "unit": FEATURE_UNITS.get(k, ""),
                    "status": res.failures.get(k, "estimated"),
                }
                for k, v in res.values.items()
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(
                "Automatic single-lead estimates. P-wave dispersion is not computed: it "
                "needs several leads. BMI and cycle data are never inferred from ECG."
            )

with tab_models:
    run = latest_run(ROOT / "runs")
    if run is None:
        st.info(
            "No experiment report found. Run `pcos-ecg demo` (synthetic) or "
            "`pcos-ecg train-evaluate --config ...` (study data) to produce one."
        )
    else:
        meta = json.loads((run / "metadata.json").read_text())
        summary = pd.read_csv(run / "summary.csv")
        if meta["data_mode"] == "synthetic_demo":
            st.error(meta["caveat"])
        else:
            st.warning(meta["caveat"])
        st.markdown(
            f"**Report:** `{run.name}` · **data mode:** {meta['data_mode']} · "
            f"**participants:** {meta['n_participants']} ({meta['n_positive_participants']} "
            f"positive) · **evaluation:** {meta['split_strategy']}, {meta['n_splits']} folds · "
            f"**seed:** {meta['seed']}"
        )
        metric = st.selectbox(
            "Metric", ["balanced_accuracy", "roc_auc", "pr_auc", "sensitivity", "specificity"]
        )
        pivot = summary.pivot(index="model", columns="condition", values=f"{metric}_mean")
        st.bar_chart(pivot)
        st.dataframe(summary, hide_index=True, width="stretch")
        if meta["warnings"]:
            st.markdown("**Warnings**")
            for w in meta["warnings"]:
                st.write("- " + w)

with tab_limits:
    st.markdown(
        """
- **Data handling.** Volunteer recordings stay in `data/private/` on the research machine
  and are git-ignored. Reports contain counts and metrics only.
- **Small cohorts.** Cross-validated scores on a few dozen participants vary a lot between
  folds and are not an external validation.
- **Synthetic balancing.** SMOTE rows are made from the training fold only and are not
  extra volunteers; test folds contain real participants only.
- **Signal processing.** Filtering is offline (zero-phase); morphology features use a
  simple threshold method on a median beat and are estimates, flagged when implausible.
- **No individual predictions.** This app does not output a PCOS risk for anyone.

See `docs/` in the repository for the data dictionary, protocol template and model card.
"""
    )
