# Claims ledger

Each CV claim, and what in this repository supports it.

| Claim | Evidence here | Status |
|---|---|---|
| Team project at RIT | – | Not verifiable from this repo (institutional record) |
| Recorded ECG and physiological data from volunteers under our own protocol | `docs/research_protocol.md`: 30 volunteers, AliveCor Kardia, departmental pilot, team-collected consent | Documented by the team; data private, so not verifiable from this repo |
| Built the ECG pipeline: preprocessing | `signal/preprocess.py`, tests in `tests/test_signal.py` | Implemented (rebuilt) |
| … waveform feature extraction | `signal/peaks.py`, `signal/features.py`, tests | Implemented (rebuilt) |
| … feature selection | `SelectKBest(mutual_info)` inside each fold, `models/factory.py` | Implemented (rebuilt) |
| … synthetic class balancing | SMOTE in `synthesis/balance.py`; leakage test in `tests/test_evaluation.py` | Implemented (rebuilt) |
| Compared RF, SVM and XGBoost with stratified evaluation | `evaluation/cv.py`, `pcos-ecg demo` report | Implemented; demo metrics only |
| For wearable-based screening | Single-lead input; offline filtering noted | Design intent; no wearable-device validation |
| Paper accepted at AIKP'26 (Paper ID 141) | – | Not verifiable from this repo; numbers in the paper come from the original code and data |

"Rebuilt" means this code was written in October 2026 from the project description,
because the original team code is not available on this machine. It follows the same
method; it is not the code that produced the paper's numbers.
