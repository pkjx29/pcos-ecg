# Project knowledge

## Confirmed facts
- Brief (from the CV): team project at RIT; volunteers recorded under the team's own
  protocol; ECG pipeline with preprocessing, waveform feature extraction, feature
  selection and synthetic class balancing; RF vs SVM vs XGBoost with stratified
  evaluation; AIKP'26 paper, ID 141.
- The original code and volunteer data are not available to this repository. This code
  is a rebuild (October 2026).

## Decisions
| Decision | Reason |
|---|---|
| SMOTE (imbalanced-learn) for "synthetic class balancing" | Standard, installed, and safe inside an imblearn `Pipeline`; SDV not installed |
| SciPy Pan–Tompkins-style detector, not NeuroKit2 | Fewer dependencies; every step is readable |
| Threshold method for wave boundaries on a median beat | Simple and testable; documented as an estimate |
| Aggregate records to participant median | Participant is the independent unit |
| Folds = min(5, minority participants // 2) | Avoids folds without positives |
| Fixed 0.5 threshold, no tuning | Small cohort; avoids optimistic bias |
| No individual predictions in the app | A demo-trained model must never look like a risk score |
| No P-wave dispersion | Needs multiple leads |

## Assumptions / open questions
- Lead II, upright R wave (detector refines to local maximum).
- 50 Hz mains (India). Change `notch_hz` elsewhere.
- Study sampling rate, device, covariates, label criteria: unknown (see
  `docs/research_protocol.md`).

## Status
All phases in `PLAN.md` done except publishing. 39 tests pass locally (Python 3.13).

## Next steps
- Fill `docs/research_protocol.md` from the team's paperwork.
- If the original code turns up, compare its features and splits with this rebuild.
- Optional: tangent-method T-end, multi-lead support.
