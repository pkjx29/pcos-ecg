# PCOS Risk Screening: Machine Learning Pipeline

## Project Overview

**Title:** Early PCOS Risk Screening using Physiological Parameters  
**Target Audience:** Biomedical Engineering Students (Minor/Undergraduate)  
**Objective:** Build an interpretable ML-based early screening tool for Polycystic Ovary Syndrome (PCOS) using physiological measurements

---

## Medical Background

### What is PCOS?
- **Polycystic Ovary Syndrome (PCOS)** is a common endocrine disorder affecting 8-20% of reproductive-age women
- Characterized by irregular menstrual cycles, elevated androgens, and/or polycystic ovaries
- Associated with metabolic dysfunction (insulin resistance, obesity)
- Risk factors: BMI, hormonal imbalance, and cardiovascular markers

### Dataset Features (Physiological Parameters)

| Feature | Unit | Description |
|---------|------|-------------|
| **HRV** | ms | Heart Rate Variability - indicates autonomic nervous system function |
| **Heart_Rate** | bpm | Resting heart rate |
| **RR_Mean** | breaths/min | Mean respiratory rate |
| **BMI** | kg/m² | Body Mass Index - metabolic health indicator |
| **BMR** | kcal/day | Basal Metabolic Rate - energy expenditure at rest |
| **Systolic_BP** | mmHg | Upper blood pressure reading |
| **Diastolic_BP** | mmHg | Lower blood pressure reading |
| **Temperature** | °C | Body temperature |
| **Age** | years | Patient age |

### Target Variable
- **PCOS**: Binary classification (0 = No PCOS, 1 = PCOS positive)

---

## Dataset Overview

### Size and Characteristics
- **Original size:** 30 samples (VERY SMALL - typical biomedical minor project)
- **Augmented size:** ~90 samples (after SMOTE + Gaussian noise)
- **Class imbalance:** Typical in real medical datasets (handled via augmentation)
- **Missing values:** Handled by removal (dropna)

**Location:** `data/pcos_data_sample.csv`

### Sample Data Format
```csv
HRV,Heart_Rate,RR_Mean,BMI,BMR,Systolic_BP,Diastolic_BP,Temperature,Age,PCOS
42.5,68,14.2,22.5,1650,118,76,37.1,22,0
35.8,75,15.1,24.3,1580,125,82,37.2,25,1
...
```

---

## Project Structure

```
PCOS_Risk_Screening/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── pcos_pipeline.py                   # Main ML pipeline script
│
├── data/
│   └── pcos_data_sample.csv          # Sample dataset
│
├── models/
│   └── best_model_*.joblib           # Saved trained model (generated)
│
└── output/
    ├── 01_eda_analysis.png           # EDA visualizations (generated)
    ├── 02_confusion_matrices.png     # Confusion matrices (generated)
    ├── 03_roc_curves.png             # ROC curves (generated)
    └── 04_feature_importance.png     # Feature importance (generated)
```

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Installation Steps

1. **Navigate to project directory:**
   ```bash
   cd PCOS_Risk_Screening
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Pipeline Workflow

The `pcos_pipeline.py` script implements a complete ML workflow with 9 stages:

### Stage 1: Data Loading and Exploration
- Load CSV dataset using pandas
- Display dataset shape, basic info, and statistics
- Identify and handle missing values
- **Output:** Cleaned dataset ready for analysis

### Stage 2: Exploratory Data Analysis (EDA)
- **Class Distribution:** Bar plot showing PCOS vs Non-PCOS samples
- **Correlation Heatmap:** Identify relationships between features
- **Feature Distributions:** Box plots for key features (Age, BMI) by PCOS status
- **Output:** `01_eda_analysis.png`

### Stage 3: Feature Preparation and Normalization
- Extract features (X) and target (y)
- Apply **StandardScaler** normalization:
  - Centers data around 0
  - Scales to unit variance (std = 1)
  - Essential for ML algorithms
- **Why?** Different features have different scales (e.g., BMI: 20-30 vs Age: 20-50)
- **Output:** Normalized feature matrix

### Stage 4: Synthetic Data Augmentation
#### Problem: Small Dataset
- Only 30 real samples → high risk of overfitting
- ML models need 100+ samples for robust training
- Solution: Synthetic data augmentation

#### Method 1: SMOTE (Synthetic Minority Over-sampling Technique)
- Creates synthetic samples by interpolating between minority class samples
- Balances dataset automatically
- **Advantage:** Mathematically principled, maintains data distribution
- **Result:** ~60 samples after balancing

#### Method 2: Gaussian Noise Augmentation
- Adds small random noise to features
- Simulates measurement variations
- Creates realistic physiological variations
- **Result:** ~90 samples total (3x augmentation)

#### Physiological Realism Checks
```python
noise_scales = {
    'HRV': 2.0,           # ±2 ms variation (typical)
    'Heart_Rate': 3.0,    # ±3 bpm variation
    'Temperature': 0.2,   # ±0.2°C variation (measurement error)
    ...
}
```

#### ⚠️ CRITICAL LIMITATIONS OF SYNTHETIC DATA

1. **Synthetic ≠ Real:** Generated samples don't represent actual patients
2. **No Deep Interactions:** Can't capture complex disease mechanisms
3. **Extrapolation Risk:** May create unrealistic edge cases
4. **Clinical Validation Required:** Must validate on real holdout patients

**Key Point:** Synthetic data is for TRAINING ONLY. Final model must be tested on real patient data.

### Stage 5: Train-Test Split
- Stratified split (maintains class distribution)
- 80% training, 20% testing
- Ensures balanced class distribution in both sets
- Random seed for reproducibility

### Stage 6: Model Training
Three models trained on augmented training data:

#### 1. **Logistic Regression**
- Linear classification model
- **Pros:** Fast, interpretable, explainable coefficients
- **Cons:** Assumes linear relationship, may underfit
- **Use Case:** Baseline model, feature importance analysis
- **Training Time:** <1 second

#### 2. **Random Forest**
- Ensemble of decision trees
- **Pros:** Handles non-linearity, feature importance, robust
- **Cons:** Less interpretable than linear models
- **Use Case:** Balanced performance, practical predictions
- **Training Time:** 1-2 seconds

#### 3. **XGBoost**
- Gradient boosting with optimized trees
- **Pros:** Often best performance, handles complex patterns
- **Cons:** More complex, requires tuning
- **Use Case:** High accuracy for screening
- **Training Time:** 1-2 seconds

### Stage 7: Model Evaluation
Comprehensive metrics for each model:

#### Classification Metrics
- **Accuracy:** (TP + TN) / Total - Overall correctness
- **Precision:** TP / (TP + FP) - False positive rate (avoid unnecessary alarms)
- **Recall:** TP / (TP + FN) - False negative rate (avoid missing cases)
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under ROC curve (0.5 = random, 1.0 = perfect)

#### Confusion Matrix
```
                Predicted
              PCOS=0  PCOS=1
Actual PCOS=0  TN     FP
       PCOS=1  FN     TP
```

**Medical Interpretation:**
- **False Negatives (FN):** WORST - missed PCOS cases
- **False Positives (FP):** Acceptable - can be confirmed by doctor
- **True Positives (TP):** GOOD - correctly identified cases
- **True Negatives (TN):** GOOD - correctly ruled out cases

### Stage 8: Visualization
Generates 4 visualizations:

1. **Confusion Matrices** - Side-by-side comparison
2. **ROC Curves** - Model discrimination ability
3. **Feature Importance** - Which features matter most
4. **Summary Statistics** - Comparison table

### Stage 9: Model Persistence
- Best model saved using joblib
- Can be loaded later for predictions
- Format: `best_model_XGBoost.joblib` (example)

---

## How to Run the Pipeline

```bash
# From project directory
python pcos_pipeline.py
```

**Expected Output:**
- Console output with detailed metrics
- 4 PNG visualization files in `output/`
- Best model saved in `models/`
- Execution time: ~10-15 seconds

---

## Key Findings and Interpretation

### Expected Results
1. **Model Performance:** ROC-AUC typically 0.80-0.95 (on augmented test data)
2. **Best Model:** Usually XGBoost or Random Forest
3. **Top Features:** Usually BMI, Heart_Rate, Age, Systolic_BP
4. **Class Balance:** Achieved 50-50 split after SMOTE

### Feature Importance Interpretation
- **High Importance Features:**
  - BMI: Obesity is strongly associated with PCOS
  - Heart_Rate: Autonomic dysfunction in PCOS
  - Systolic_BP: Cardiovascular risk elevation
  
- **Low Importance Features:**
  - Temperature: General health marker, less PCOS-specific
  - Age: Age is included but normalized

### Model Selection Criteria
1. **Best ROC-AUC:** Primary metric for screening
2. **High Recall:** Minimize false negatives (missed cases)
3. **Acceptable Precision:** Minimize unnecessary referrals
4. **Interpretability:** Prefer simpler models if performance similar

---

## Important Limitations and Ethical Considerations

### 1. **Synthetic Data Limitations**
- ✗ Augmented data is for training robustness only
- ✗ Does NOT replace real patient data collection
- ✗ Models trained on 90 samples with 67% synthetic data
- ✓ Real validation dataset needed before clinical use

### 2. **Small Sample Size**
- Original 30 samples is insufficient for production ML
- Typical medical studies need 200-1000+ samples
- This project demonstrates methodology, not clinical deployment

### 3. **Dataset Bias**
- Unknown demographic composition
- May not generalize to different populations
- Require representative data collection

### 4. **Not a Diagnostic Tool**
- ✗ NOT meant to diagnose PCOS
- ✓ Screening/early detection indicator only
- ✓ Must be validated by endocrinologist
- ✓ Should supplement, not replace, clinical diagnosis

### 5. **Regulatory Compliance**
- For production use: Requires regulatory approval (FDA, CE marking)
- Clinical validation studies needed
- Patient privacy protection essential
- Transparent documentation of limitations required

### 6. **Medical Ethics**
- Avoid false negatives (missed cases) → Prioritize recall
- Communicate uncertainty to patients
- Maintain human-in-the-loop decision making
- Regular model revalidation on new data

---

## Advanced Topics for Extension

If you want to extend this project:

1. **Hyperparameter Tuning**
   - Use GridSearchCV for optimal parameters
   - Cross-validation for robust evaluation

2. **Feature Engineering**
   - Create interaction terms (e.g., BMI × Age)
   - Non-linear transformations
   - Domain-specific ratios

3. **Deep Learning** (if more data available)
   - Neural networks for non-linear patterns
   - Requires 1000+ samples

4. **Explainability**
   - SHAP values for model interpretation
   - LIME for local explanations
   - Feature effect plots

5. **Production Deployment**
   - REST API using Flask/FastAPI
   - Model monitoring and retraining
   - A/B testing for model updates

6. **Data Collection**
   - Recruit real patients
   - Multi-center validation studies
   - Longitudinal follow-up

---

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'xgboost'`
```bash
Solution: pip install xgboost
```

**Issue:** Data file not found
```bash
Solution: Ensure pcos_data_sample.csv exists in data/ folder
```

**Issue:** Low model performance (ROC-AUC < 0.70)
```bash
Solutions:
- Collect more real data
- Engineer better features
- Adjust SMOTE parameters (k_neighbors)
- Tune model hyperparameters
```

**Issue:** Memory error
```bash
Solution: Reduce augmentation factor (line 205-210 in pcos_pipeline.py)
```

---

## References and Further Reading

### PCOS
1. Witchel, S. F., Oberfield, S. E., & Peña, A. S. (2020). Polycystic ovary syndrome: Pathophysiology, presentation, and treatment with focus on adolescent girls. Journal of the Endocrine Society, 4(2), bvz028.

### ML Methods
2. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR, 16, 321-357.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.

### Medical ML Ethics
4. FDA-led AI/ML Action Plan (2021): https://www.fda.gov/news-events/press-announcements/
5. IEEE Ethically Aligned Design: https://ethicsinaction.ieee.org/

### Code Resources
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

---

## Contributors

**Course:** Biomedical Engineering Minor - College Project  
**Academic Year:** 2025  
**Supervisor:** [Faculty Name]

---

## License

This project is provided for educational purposes. Use responsibly in academic settings.

**Disclaimer:** This is a demonstration project. Do NOT use for clinical diagnosis without proper validation and regulatory approval.

---

## Summary Checklist

- [x] Complete data pipeline from loading to evaluation
- [x] Proper handling of small dataset (augmentation)
- [x] Medical parameter validation
- [x] Multiple models for comparison
- [x] Comprehensive metrics and visualizations
- [x] Interpretable ML methods (no deep learning)
- [x] Beginner-friendly code with detailed comments
- [x] Production-ready model saving
- [x] Discussion of limitations and ethics
- [x] Documentation and examples

---

**Questions?** Review the detailed comments in `pcos_pipeline.py` for implementation details.

**Ready to learn?** Start with `README.md`, then examine `pcos_pipeline.py` section by section.
