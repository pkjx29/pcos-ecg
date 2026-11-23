# Claims ledger

Each CV claim, and what in this repository supports it. "Original study" means the team's
notebooks in [`original_study/`](../original_study/); "rebuild" means the tested package in
`src/`.

| Claim | Original study | Rebuild | Status |
|---|---|---|---|
| Team project at RIT | Credited in `original_study/README.md` | – | Institutional record; not verifiable from code |
| Recorded ECG and physiological data from volunteers under our own protocol | Notebooks expect the private volunteer table | `docs/research_protocol.md`: 30 volunteers, AliveCor Kardia, departmental pilot, team-collected consent | Documented by the team; data private |
| Built the ECG pipeline: preprocessing | **None.** ECG intervals were estimated with AI assistance from the Kardia printouts | `signal/preprocess.py`, tests in `tests/test_signal.py` | Rebuild only; NeuroKit2 re-extraction of the real recordings planned |
| … waveform feature extraction | **None** (as above) | `signal/peaks.py`, `signal/features.py`, tests | Rebuild only |
| … feature selection | `03_3_Feature_Selection_Boruta.ipynb`: correlation filter, mutual information, Boruta | `SelectKBest(mutual_info)` inside each fold | Both; the original ran selection on the full dataset |
| … synthetic class balancing | `03_1` synthesizer comparison and `03_2` 30 → 150 synthesis (a class-conditional Gaussian) | SMOTE inside training folds only, with a leakage test | Both; the original augmented before splitting |
| Compared RF, SVM and XGBoost with stratified evaluation | `03_4_Model_Comparison_RF_SVM_XGB.ipynb`; `04_Results/04_Model_Comparison/model_comparison.csv` | `evaluation/cv.py`, `pcos-ecg demo` | Both; original results are on synthetic data only |
| For wearable-based screening | ECG-only Random Forest proposed as the deployment model | Single-lead input; offline filtering noted | Design intent; no wearable-device validation |
| Paper accepted at AIKP'26 (Paper ID 141) | Cited in `original_study/README.md` | – | Not verifiable from code |

## What the original numbers mean

The paper's AUC of 0.95 comes from one 80:20 split; the cross-validated ROC-AUC for the
ECG-only Random Forest is 0.922. Both were trained and tested only on the 150 synthetic rows,
so they describe the synthesizer more than real participants. See the caveats in
[`original_study/README.md`](../original_study/README.md).
