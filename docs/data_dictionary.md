# Data dictionary

All ECG features are computed per recording from one lead, then summarised per
participant (median) before modelling. "Estimate" means an automatic measurement that
has not been checked against expert annotation.

## ECG-derived features

| Column | Unit | Definition | Source | Plausible range | Notes |
|---|---|---|---|---|---|
| `n_beats` | count | R peaks detected | waveform | – | Excluded from predictors |
| `mean_hr_bpm` | beats/min | 60 / mean cleaned RR | waveform | – | |
| `mean_rr_s` | s | mean cleaned RR interval | waveform | 0.3–2.0 per interval | RR within 20% of median; neighbours of rejected RR dropped |
| `hrv_sdnn_estimate_s` | s | SD of cleaned RR (ddof 1) | waveform | – | Short-term only; recordings are short |
| `hrv_rmssd_s` | s | RMS of successive RR differences | waveform | – | |
| `sdnn_meanrr_ratio` | – | SDNN / mean RR | waveform | – | Coefficient of variation of RR |
| `pr_interval_s` | s | P onset → QRS onset on median beat | waveform | 0.08–0.30 | Estimate; NaN if P not above noise |
| `qrs_duration_s` | s | QRS onset → offset (5% of R amplitude) | waveform | 0.04–0.20 | Estimate |
| `qt_interval_s` | s | QRS onset → T end (10% of T amplitude) | waveform | 0.20–0.60 | Threshold method, not tangent method |
| `qtc_bazett_s` | s | QT / √(mean RR) | waveform | – | Bazett over-corrects at high heart rates |
| `t_wave_duration_s` | s | T onset → T end (10% threshold) | waveform | 0.05–0.40 | Estimate |
| `r_amplitude_mv` | mV | R peak minus PR-segment level | waveform | – | Depends on electrode placement |

**Not computed:** P-wave dispersion (max − min P duration across leads). A single lead
cannot support it, so no single-lead stand-in is reported under that name.

## Record bookkeeping (never predictors)

`subject_id`, `record_id`, `quality_ok`, `n_beats`, `label`.

## Covariates (optional, separate source)

Examples: `BMI`, `Cycle_Length`, `Cycle_Regularity`, `Family_History`. They must come from
a documented measurement or questionnaire. They are never inferred from the ECG.

| Field | Status |
|---|---|
| Which covariates the RIT study collected | not yet documented |
| Units and collection method | not yet documented |

## Label

| Field | Status |
|---|---|
| Meaning of positive class | Self-reported PCOS (1) vs no PCOS reported (0) |
| Label level | participant |
| Who assigned the label, by which criteria | The volunteer, by self-report; not clinically confirmed |
| Handling of uncertain diagnoses | not yet documented |

## Provenance

| Field | Value |
|---|---|
| Data source | RIT departmental pilot, 30 volunteers (study data); synthetic generator (demo) |
| Consent / approval status | Consent collected by the project team; departmental permission; no ethics committee review for the pilot (see `research_protocol.md`) |
| Acquisition device | AliveCor Kardia |
| Leads, sampling rate | pending (single-lead Kardia records lead I) |
| Known exclusions | not yet documented |
| Synthetic-data method | demo: Gaussian-sum beat model (`demo/synthetic.py`); balancing: SMOTE on training folds |
