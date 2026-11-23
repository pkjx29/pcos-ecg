# Project knowledge

## Confirmed facts
- Brief (from the CV): team project at RIT; volunteers recorded under the team's own
  protocol; ECG pipeline with preprocessing, waveform feature extraction, feature
  selection and synthetic class balancing; RF vs SVM vs XGBoost with stratified
  evaluation; AIKP'26 paper, ID 141.
- The team's original notebooks are in `original_study/` (published November 2026 with the
  team's permission). The ECG intervals in the study were estimated with AI assistance,
  not extracted by code. Since v0.2.0 the main pipeline follows the study's method with
  leakage removed (see README, "Pipeline").
- Pilot facts from the team: 30 volunteers, AliveCor Kardia, PCOS self-reported, consent
  collected by the team, departmental permission, no ethics committee review for the
  pilot (application submitted for the larger study), coded IDs with the key held by the
  team. Details in `docs/research_protocol.md`.

## Decisions
| Decision | Reason |
|---|---|
| Study's class-conditional Gaussian for synthetic expansion; SDV optional; SMOTE baseline only | Matches original_study/03_2; honest name |
| Synthesis and Boruta selection per training fold | The study did both on all data, leaking test information |
| Leave-one-subject-out for study data | About 30 participants; every person tested once |
| NeuroKit2 delineation by default; scipy detector kept | Planned path for the real recordings; scipy has no P-wave boundaries |
| Threshold method for wave boundaries on a median beat | Simple and testable; documented as an estimate |
| Aggregate records to participant median | Participant is the independent unit |
| Folds = min(5, minority participants // 2) | Avoids folds without positives |
| Fixed 0.5 threshold, no tuning | Small cohort; avoids optimistic bias |
| No individual predictions in the app | A demo-trained model must never look like a risk score |
| P-wave dispersion as a single-lead proxy (beat-to-beat P-duration spread) | Keeps the study's feature; true dispersion needs multiple leads |

## Assumptions / open questions
- Lead I (single-lead Kardia), upright R wave (detector refines to local maximum).
- 50 Hz mains (India). Change `notch_hz` elsewhere.
- Kardia model, sampling rate and covariates: unknown (see `docs/research_protocol.md`).

## Status
v0.2.0 on branch `original-method-rebuild`: 73 tests pass locally (Python 3.13, project `.venv`).

## Next steps
- Fill `docs/research_protocol.md` from the team's paperwork.
- Run NeuroKit2 extraction on the real Kardia recordings and compare with the study's values.
- Optional: tangent-method T-end, multi-lead support.
