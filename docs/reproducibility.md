# Reproducibility

## Environment
Python ≥ 3.10 (tested locally on 3.13, CI on 3.11 and 3.12).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

`metadata.json` in each run records the Python and package versions actually used.

## Seeds
| Where | Setting | Default |
|---|---|---|
| CV splits, models, SMOTE, mutual information | `seed` in experiment config | 42 |
| Synthetic demo cohort | `pcos-ecg demo --seed` | 7 |

## Commands
```bash
pcos-ecg demo                                             # synthetic, end to end
pcos-ecg validate-data    --config configs/local_data.yaml
pcos-ecg extract-features --config configs/local_data.yaml
pcos-ecg train-evaluate   --config configs/local_experiment.yaml
```

## Artifacts
| Output | Location | Contains |
|---|---|---|
| Feature table | `output.features_path` | participant IDs and features: private for study data |
| QC report | `output.qc_report_path` | counts only |
| Experiment run | `runs/<UTC time>_<data_mode>/` | metadata, per-fold and summary metrics, Markdown report |

## Supplied locally, never committed
Waveform, label and covariate CSVs in `data/private/`, and `configs/local_*.yaml`.
