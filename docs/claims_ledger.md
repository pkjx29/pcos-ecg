# Claims ledger

Each CV claim, and what in this repository supports it. "Original study" means the team's
notebooks in [`original_study/`](../original_study/); "rebuild" means the tested package in
`src/`.

| Claim | Original study | Rebuild | Status |
|---|---|---|---|
| Team project at RIT | Credited in `original_study/README.md` | – | Institutional record; not verifiable from code |
| Recorded ECG and physiological data from volunteers under our own protocol | Notebooks expect the private volunteer table | `docs/research_protocol.md`: 30 volunteers, AliveCor Kardia, departmental pilot, team-collected consent | Documented by the team; data private |
| Built the ECG pipeline: preprocessing | **None.** ECG intervals were estimated with AI assistance from the Kardia printouts | NeuroKit2 cleaning (`signal/nk_features.py`); scipy band-pass alternative (`signal/preprocess.py`) | Rebuild only; to be run on the real Kardia recordings |
| … waveform feature extraction | **None** (as above) | NeuroKit2 R peaks and wavelet delineation with per-beat plausibility checks; the study's feature names and engineered features (`features/study.py`) | Rebuild only; P-wave dispersion is a single-lead proxy |
| … feature selection | `03_3_Feature_Selection_Boruta.ipynb`: correlation filter, mutual information, Boruta on the whole dataset | Same three stages, run inside each training fold (`features/selection.py`) | Both; the rebuild removes the leakage |
| … synthetic class balancing | `03_1` synthesizer comparison; `03_2` 30 → 150 expansion (a class-conditional Gaussian) on all participants | Same comparison and same generator, fitted on each fold's training participants (`synthesis/`); SMOTE and class weights as labelled baselines | Both; the rebuild removes the leakage |
| Compared RF, SVM and XGBoost with stratified evaluation | `03_4`: 9 experiments; `04_Results/04_Model_Comparison/model_comparison.csv` | Same 9 plus the 0.7/0.3 ensemble, tested on held-out real participants (leave-one-subject-out for study data) | Both; original results are on synthetic data only, rebuild results so far are demo-only |
| For wearable-based screening | ECG-only Random Forest proposed as the deployment model | Single-lead input; offline filtering noted | Design intent; no wearable-device validation |
| Paper accepted at AIKP'26 (Paper ID 141) | Cited in `original_study/README.md` | – | Not verifiable from code |

## What the original numbers mean

The paper's AUC of 0.95 comes from one 80:20 split; the cross-validated ROC-AUC for the
ECG-only Random Forest is 0.922. Both were trained and tested only on the 150 synthetic rows,
so they describe the synthesizer more than real participants. See the caveats in
[`original_study/README.md`](../original_study/README.md).
