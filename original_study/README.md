# Original study: notebooks and aggregate results

These are the notebooks the team wrote for the original study, published with the team's
permission, together with their aggregate outputs.

- **Project:** Early Screening of PCOS from ECG Waveform Features and Physiological Parameters
- **Context:** 6th-semester mini-project, Department of Medical Electronics Engineering, Ramaiah
  Institute of Technology (RIT), Bengaluru, 2025–26, guided by Dr. Budaraju Dhananjay
- **Team:** Hamsa M L, Harini C M, Priyanshu Kumar Jha, T R Eeksha Patel
- **Paper:** "Early Screening of PCOS from ECG Waveform Features and Physiological Parameters",
  AIKP'26, Paper ID 141. The paper itself is not included here.

The rest of this repository is a separate, later rebuild of the method with stricter evaluation.
This folder shows what was actually run for the paper.

## These notebooks will not run as published

They expect the volunteers' data, which stays private and is not in this repository. Outputs
have been cleared from every notebook, and Google Drive paths have been removed.

## Pipeline order

| Step | File | What it does |
|---|---|---|
| 1. Synthesizer comparison | `03_Code/03_1_Synthesizer_Comparison_10to30.ipynb` | Trains CTGAN, Gaussian Copula and a custom Bayesian-network synthesizer on 10 subjects, generates 30 rows from each, and scores them with SDMetrics, KS and correlation error |
| 2. 30 → 150 synthesis | `03_Code/03_2_Synthesis_30to150.py` | Generates the 150-row dataset used by every later step. Transcribed from the Colab notebook `notusing_10_>150`, whose file name could not be downloaded |
| 3. Feature selection | `03_Code/03_3_Feature_Selection_Boruta.ipynb` | Correlation filter (\|r\| > 0.90), mutual information, Boruta; PCA/UMAP/t-SNE; Mann–Whitney U tests |
| 4. Model comparison | `03_Code/03_4_Model_Comparison_RF_SVM_XGB.ipynb` | Random Forest, XGBoost and SVM on ECG-only, physiology-only and combined features (9 experiments); 5-fold stratified CV; SHAP |

Other files: `03_2b`, `03_3b`, `03_4b` and `03_5` are alternative versions of steps 2–4, and
`03_0_Early_Versions/` holds earlier drafts. The early template uses made-up sample numbers,
not volunteer data.

## Results

`04_Results/` holds only aggregate outputs: metric tables, SHAP importances, feature-selection
tables, synthesizer scores, and plots. No file contains a row per person.

Headline numbers from `04_Results/04_Model_Comparison/model_comparison.csv` (5-fold stratified CV):

| Model | Features | ROC-AUC |
|---|---|---|
| Random Forest | ECG only | 0.922 |
| Random Forest | Combined | 0.915 |
| XGBoost | ECG only | 0.902 |
| SVM (RBF) | ECG only | 0.826 |
| All three | Physiology only | 0.68–0.73 |

The paper's AUC of 0.95 and its confusion matrix (14 / 2 / 1 / 13) come from a single 80:20 split.

## Read these results with three caveats

1. **The ECG intervals were not extracted by code.** The values in the study's feature table
   were estimated with AI assistance from the Kardia printouts. No signal-processing script
   exists for the original study. A re-extraction with NeuroKit2 is planned.
2. **The headline AUC was trained and tested on synthetic data.** Every row used for training
   and testing comes from the 150-row synthetic dataset; no real volunteer was held out for
   testing. Because each class is generated from its own distribution, the classes are easier
   to separate than real participants would be, so the scores mostly describe the generator.
3. **The 30 → 150 "Bayesian network" is a class-conditional Gaussian.** The class label is
   drawn at random, then the numeric features are sampled from a multivariate normal
   distribution fitted to that class. It is not a learned Bayesian network.

The synthesizer comparison score of 0.9355 comes from the 10 → 30 experiment. The 150-row
dataset itself scored 0.785 and did not meet the original certification thresholds
(see `04_Results/04_Synthesis_Validation/dataset150_certification_report.csv`).

The rebuild in the rest of this repository addresses the evaluation issues: participants are
grouped in cross-validation, synthetic rows are generated from training folds only, and
features come from code. See [`../docs/claims_ledger.md`](../docs/claims_ledger.md).
