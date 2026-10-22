# Data

This repository contains **no participant data**, and must never contain any.

| Path | What goes there | Committed? |
|---|---|---|
| `data/private/` | Real study files: waveforms, labels, covariates, extracted features, QC reports | No (git-ignored) |
| `data/demo/` | Output of `pcos-ecg demo`: synthetic recordings of made-up people | No (regenerated on demand) |

## Expected study files

Column names are mapped in `configs/local_data.yaml` (copy from
`configs/data_schema.example.yaml`), so your files do not need these exact names.

**Waveforms** (`ecg_long.csv`), one row per sample:

| column | meaning |
|---|---|
| `subject_id` | pseudonymous participant code (no names, phone numbers or dates of birth) |
| `record_id` | one recording session; belongs to exactly one participant |
| `lead` | e.g. `II` |
| `sample_index` | 0, 1, 2 … within the record |
| `sample_rate_hz` | constant within a record |
| `amplitude` | in mV (or µV with `amplitude_unit: uV`) |

**Labels** (`labels.csv`): `subject_id`, label column. One label per participant.

**Covariates** (optional): `subject_id` plus measured or self-reported fields such as BMI.
These are never derived from the ECG.

Why not commit raw data: ECG waveforms are health data and can act as a biometric
identifier, and the volunteers' consent covered the study, not public release. See the
README's "Data protection" section.
