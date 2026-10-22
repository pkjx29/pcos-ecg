# Model card

**Models:** majority-prior baseline, Random Forest, RBF-kernel SVM, XGBoost; each with
three balancing conditions (none, class weights, SMOTE on training folds).

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
Participant-grouped stratified k-fold (k ≤ 5, limited by the number of minority-class
participants); fixed 0.5 threshold; fixed hyperparameters; same folds for every model and
condition. No separate locked test set is used, because the cohort is too small to hold
one out; this is internal validation only.

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
- SMOTE interpolates in feature space and can produce physiologically odd combinations.
- Bazett correction is biased at high heart rates.

## Version history
| Version | Date | Change |
|---|---|---|
| 0.1.0 | 2026-10-22 | Rebuilt public pipeline with synthetic demo |
