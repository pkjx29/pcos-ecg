# Data dictionary

Column names follow the original study's notebooks. ECG features are computed per
recording from one lead, then summarised per participant (median of quality-checked
recordings). "Estimate" means an automatic measurement not checked against expert
annotation. The default extraction backend is NeuroKit2 (`signal/nk_features.py`); the
built-in "scipy" backend measures the same intervals except P-wave dispersion.

## ECG features (measured)

| Column | Unit | Definition (NeuroKit2 backend) | Plausible range per beat | Notes |
|---|---|---|---|---|
| `HRV_SDNN_Estimate` | ms | SD of cleaned RR intervals (ddof 1) | – | Short-term only; recordings are about 30 s |
| `Mean_RR_Interval` | s | Mean cleaned RR interval | 0.3–2.0 | RR within 20% of median; neighbours of a rejected RR dropped |
| `PR_Interval` | s | P onset → QRS onset, median over beats | 0.08–0.30 | Wavelet delineation |
| `QRS_Duration` | s | QRS onset → offset, median over beats | 0.04–0.20 | |
| `QT_Interval` | s | QRS onset → T offset, median over beats | 0.20–0.60 | |
| `QTc_Bazett` | s | QT / √(mean RR) | – | Bazett over-corrects at high heart rates |
| `T_Wave_Duration` | s | T onset → T offset, median over beats | 0.05–0.40 | |
| `P_Wave_Dispersion` | s | **Single-lead proxy:** 95th − 5th percentile of P-wave duration across beats | P duration 0.04–0.20 | Classical P-wave dispersion is max − min P duration across 12 leads and cannot be measured from one lead. This is a different quantity, kept under the study's name so results line up |

A value is NaN with a recorded reason when too few beats give a plausible measurement.

## Engineered features (row-wise formulas, report section 3.5.4)

| Column | Formula | Unit |
|---|---|---|
| `SDNN_MeanRR_Ratio` | (SDNN in s) / Mean RR | unitless (coefficient of variation of RR) |
| `QTc_MeanRR_Ratio` | QTc / Mean RR | 1/s |
| `PWaveDisp_x_QTc` | P_Wave_Dispersion × QTc | s² |
| `HRVSDNN_x_QTc` | SDNN (ms) × QTc | ms·s |

The report does not state the units it used for SDNN / Mean RR; here both are in seconds.

## The study's final 10 features (feature sets in the experiment config)

- **ECG (6):** HRV_SDNN_Estimate, P_Wave_Dispersion, SDNN_MeanRR_Ratio, QTc_Bazett,
  PR_Interval, T_Wave_Duration
- **Physiological (4):** Cycle_Length, Cycle_Regularity, BMI, Family_History

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
| Synthetic-data method | demo: Gaussian-sum beat model and made-up covariates (`demo/synthetic.py`); expansion: class-conditional Gaussian on each fold's training participants (SDV Gaussian Copula or CTGAN optional); SMOTE only as a baseline |
