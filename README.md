# Early Screening of PCOS from ECG Waveforms

ECG processing pipeline and leakage-safe model comparison (Random Forest, SVM, XGBoost)
for research on wearable-based screening of polycystic ovary syndrome.

> **Research prototype.** Not a medical device and not a diagnostic tool. Nothing in this
> repository has been clinically validated.

## Background

This was a team project at RIT Bengaluru (Medical Electronics). In a pilot study in the
departmental lab, the team recorded ECG (AliveCor Kardia) and other physiological data
from 30 volunteers under its own protocol, with PCOS status self-reported. It built an ECG pipeline
(preprocessing, waveform feature extraction, feature selection) with synthetic class
balancing, and compared Random Forest, SVM and XGBoost with stratified evaluation. The
paper was accepted at AIKP'26 (Paper ID 141).

## What is in this repository, and what is not

| | |
|---|---|
| ✅ The full pipeline, rebuilt as a tested Python package | ❌ Volunteer recordings or labels |
| ✅ A synthetic ECG generator so the pipeline runs end to end | ❌ The paper's results |
| ✅ Templates for documenting the protocol and data | ❌ Any trained model |

**The code here is a rebuild.** The original team code is not available, so this
repository re-implements the method from the project description. It is the same
approach, written again with tests. It is not the code that produced the paper's
numbers, and running it on the demo data does not reproduce them. See
[`docs/claims_ledger.md`](docs/claims_ledger.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pcos-ecg demo              # synthetic cohort → features → model comparison report
pytest                     # 39 tests, synthetic data only
streamlit run app/Home.py  # dashboard
```

`pcos-ecg demo` generates 30 made-up participants, the same size as the pilot (an arbitrary 10 in the positive class, two 30 s
recordings each at 250 Hz), extracts features and writes a report to `runs/`. The class
difference in the generator is arbitrary, so **the demo metrics say nothing about PCOS**.

## Pipeline

```mermaid
flowchart LR
    A[Single-lead ECG] --> B[Quality checks]
    B --> C[Band-pass 0.5–40 Hz<br/>+ 50 Hz notch]
    C --> D[R-peak detection]
    D --> E[HRV + interval features<br/>median beat]
    E --> F[Per-participant table]
    F --> G[Grouped stratified CV]
    G --> H["Per fold: impute → scale →<br/>select k best → SMOTE → model"]
    H --> I[Report]
```

1. **Quality checks**: reject empty, short, non-finite, out-of-range, saturated or flat
   recordings, with the reason recorded.
2. **Filtering**: 4th-order Butterworth band-pass and a 50 Hz notch, applied forward and
   backward (zero phase). This is offline processing; a wearable would need a causal filter.
3. **R peaks**: simplified Pan–Tompkins (5–15 Hz band, derivative, squaring, 150 ms
   integration, 250 ms refractory period), then RR cleaning.
4. **Features**: heart rate, SDNN, RMSSD, SDNN/mean RR, and PR, QRS, QT, QTc (Bazett) and
   T-wave duration from a median beat. A feature that cannot be measured is left empty
   with a reason; it is never filled with a guess. P-wave dispersion needs several leads
   and is not computed.
5. **Evaluation**: participants, not recordings, are the unit. `StratifiedGroupKFold`
   keeps each person in one fold. Imputation, scaling, mutual-information feature
   selection and SMOTE are all fitted inside each training fold, so test folds contain
   only real, unseen participants. Each model runs under three conditions: no
   balancing, class weights, and SMOTE.
6. **Report**: sensitivity, specificity, PPV, F1, balanced accuracy, ROC-AUC and PR-AUC
   per fold, with mean and SD across folds.

Details: [`docs/architecture.md`](docs/architecture.md),
[`docs/data_dictionary.md`](docs/data_dictionary.md).

## Running on study data

Study files go in `data/private/` (git-ignored); see [`data/README.md`](data/README.md).

```bash
cp configs/data_schema.example.yaml configs/local_data.yaml        # edit column names
cp configs/experiment.example.yaml  configs/local_experiment.yaml
pcos-ecg validate-data    --config configs/local_data.yaml
pcos-ecg extract-features --config configs/local_data.yaml
pcos-ecg train-evaluate   --config configs/local_experiment.yaml
```

These commands never fall back to demo data: a missing file is an error. If the
smaller class has too few participants for at least two folds, `train-evaluate` stops
and says the comparison cannot be estimated reliably.

## Data protection

How the pilot data were handled:

- **Consent and permission.** The project team collected consent from each volunteer, and
  the pilot ran in the departmental lab with the competent authority's permission. It
  did not go through an ethics committee. A review application has been submitted for
  the larger follow-on study.
- **Coded data.** Volunteers appear in the data files under codes, not names. Only the
  project team holds the list linking codes to people.
- **Volunteer data stays private.** Raw ECG is health data and can identify a person, and
  participants agreed to a study, not to publication. Nothing from the recordings
  (waveforms, features, labels, trained models) is in this repository.

How this repository enforces it:

- **Pseudonymous IDs only.** The pipeline expects participant codes, not names or dates
  of birth.
- **Reports are aggregate.** QC and experiment reports contain counts and metrics only,
  never IDs or rows; a test checks this.
- **Ignored paths.** `data/private/`, local configs, `runs/` and model files are
  git-ignored, and a test fails if any data file is tracked.
- **Synthetic stand-in.** The public demo uses generated signals, labelled as synthetic
  wherever they appear.
- **Documentation gaps are marked.** The protocol is recorded in
  [`docs/research_protocol.md`](docs/research_protocol.md); details not yet documented
  stay "pending" rather than being guessed.

## Repository layout

```
src/pcos_ecg/   ingestion/ signal/ features/ synthesis/ models/ evaluation/ reporting/ demo/ cli.py
app/Home.py     Streamlit dashboard (demo-safe; no individual predictions)
configs/        example and demo configs (no private paths)
tests/          pytest suite on synthetic data
docs/           architecture, data dictionary, protocol template, model card, deployment, claims
```

## Limitations

- PCOS status in the pilot was self-reported, not clinically confirmed.
- With 30 participants, cross-validated scores are unstable, and internal CV is not
  external validation.
- Morphology features are automatic single-lead estimates, not expert annotations.
- SMOTE interpolates between participants in feature space; the synthetic rows are not
  extra volunteers.
- Bazett's QTc correction is biased at high heart rates.

## License

MIT. See [LICENSE](LICENSE).
