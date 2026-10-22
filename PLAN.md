# Plan

Status of each phase. A box is ticked only after its checks ran and passed.

- [ ] **0. Audit** – no original code or study data on this machine; rebuild from the CV brief; synthetic demo data only.
- [ ] **1. Scaffold** – package, configs, `.gitignore`, CLI `--help`.
- [ ] **2. Data contracts** – waveform + label loaders, schema/cohort validation, privacy-safe QC report.
- [ ] **3. Signal + features** – quality checks, zero-phase filtering, R-peak detection, interval/HRV features with failure reasons.
- [ ] **4. Balancing** – SMOTE inside training folds only; class-weight comparator; minimum-data guard.
- [ ] **5. Evaluation** – participant-grouped stratified CV, RF / SVM / XGBoost / baseline, fold-local selection, reports.
- [ ] **6. App** – Streamlit dashboard reading only produced reports; demo mode labelled.
- [ ] **7. Tests + CI** – pytest, ruff, GitHub Actions.
- [ ] **8. Docs** – README, knowledge.md, CLAUDE.md, docs/.
- [ ] **9. Publish** – GitHub repo created and pushed (with owner approval).

## Blocked on study data / owner input
- Real recordings, labels and covariates (`data/private/`, never committed).
- Acquisition protocol, device, leads, sample rate, consent and approval details (`docs/research_protocol.md`).
- Paper numbers: none are reproduced here; see `docs/claims_ledger.md`.
