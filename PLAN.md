# Plan

Status of each phase. A box is ticked only after its checks ran and passed.

- [x] **0. Audit**: no original code or study data on this machine; rebuild from the CV brief; synthetic demo data only.
- [x] **1. Scaffold**: package, configs, `.gitignore`, CLI `--help`.
- [x] **2. Data contracts**: waveform + label loaders, schema/cohort validation, privacy-safe QC report.
- [x] **3. Signal + features**: quality checks, zero-phase filtering, R-peak detection, interval/HRV features with failure reasons.
- [x] **4. Synthesis**: the study's class-conditional Gaussian (plus SDV Gaussian Copula / CTGAN) on training participants only; SMOTE and class weights as baselines.
- [x] **5. Evaluation**: participant-grouped stratified CV, RF / SVM / XGBoost / baseline, fold-local selection, reports.
- [x] **6. App**: Streamlit dashboard reading only produced reports; demo mode labelled.
- [x] **7. Tests + CI**: pytest and ruff pass locally; GitHub Actions workflow written (runs after first push).
- [x] **8. Docs**: README, knowledge.md, CONTRIBUTING.md, docs/.
- [x] **9. Publish**: public repo github.com/pkjx29/pcos-ecg created and pushed with owner approval.

## Blocked on study data / owner input
- Real recordings, labels and covariates (`data/private/`, never committed).
- Acquisition protocol, device, leads, sample rate, consent and approval details (`docs/research_protocol.md`).
- Paper numbers: none are reproduced here; see `docs/claims_ledger.md`.
