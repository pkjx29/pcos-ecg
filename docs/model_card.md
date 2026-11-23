# Model card

**Models:** Random Forest, XGBoost and RBF-kernel SVM on ECG-only, physiology-only and
combined features (the study's 9 experiments), a 0.7 ECG / 0.3 physiology soft-vote
ensemble, and a prior-only baseline. Main condition: the study's synthetic expansion,
fitted on each fold's training participants. Baselines: real data only, class weights,
and SMOTE on training participants.

## Intended use
Research comparison of classifiers on ECG-derived features, for a student project on
wearable-oriented PCOS screening.

## Out of scope
Diagnosis, triage, or any decision about a real person. No model in this repository is
validated for clinical use, and ECG alone is not an established basis for diagnosing PCOS.

## Data
- **Public repository:** synthetic demo data only. Metrics from it describe the generator.
- **Study data (RIT pilot):** 30 volunteers recorded with an AliveCor Kardia in the
  departmental lab; PCOS status self-reported. Not included in this repository. See
  `research_protocol.md`.

## Evaluation design
Leave-one-subject-out for study data (participant-grouped stratified k-fold for the
demo). Synthetic rows and feature selection come from the training participants of each
fold only, so every test prediction is for a real, held-out participant. Metrics are
computed on the pooled out-of-fold predictions. Fixed 0.5 threshold, fixed
hyperparameters, same folds for every model and condition. No separate locked test set,
because the cohort is too small to hold one out; this is internal validation only.

This differs from the original study, which trained and tested on synthetic rows and
selected features on the whole dataset (see `original_study/README.md`).

## Metrics
Only from report files in `runs/`. None are reproduced in this card. Paper results
(AIKP'26, Paper ID 141) come from the original study code and data, not from this
repository.

## Known limitations
- Labels are self-reported, not clinically confirmed; scores measure agreement with
  self-report.
- Small cohort (30-participant pilot): fold-to-fold variance is large; no confidence
  intervals are claimed.
- Single-lead automatic morphology estimates; no expert annotation.
- Synthetic rows (class-conditional Gaussian, or SMOTE in the baseline) are not extra
  volunteers and can produce physiologically odd combinations.
- P-wave dispersion is a single-lead proxy, not the 12-lead quantity.
- Demo metrics describe the synthetic generator and say nothing about PCOS.
- Bazett correction is biased at high heart rates.

## Version history
| Version | Date | Change |
|---|---|---|
| 0.1.0 | 2026-10-22 | Rebuilt public pipeline with synthetic demo |
| 0.2.0 | 2026-11-23 | Follows the original study's method (synthesis, Boruta selection, 9 experiments, ensemble, NeuroKit2), evaluated on held-out real participants |
