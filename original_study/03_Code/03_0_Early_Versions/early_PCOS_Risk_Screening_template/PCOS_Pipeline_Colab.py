# ========================================================================================================
# PCOS RISK SCREENING - COMPLETE ML PIPELINE FOR GOOGLE COLAB
# ========================================================================================================
# This is a complete, runnable Python script for Google Colab
# Simply copy-paste this entire file into a Google Colab notebook cell or upload as notebook
# ========================================================================================================

# !pip install scikit-learn xgboost imbalanced-learn pandas numpy matplotlib seaborn joblib -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 100)
print("PCOS RISK SCREENING - COMPLETE ML PIPELINE")
print("=" * 100)

# ========================================================================================================
# STEP 1: CREATE SAMPLE DATASET (or use your own CSV)
# ========================================================================================================
print("\n[STEP 1: CREATING SAMPLE DATASET]")
print("-" * 100)

# Create sample data inline
data = {
    'HRV': [42.5, 35.8, 51.2, 38.5, 48.3, 32.1, 55.7, 36.9, 44.2, 33.4,
            49.8, 37.6, 52.1, 34.9, 46.5, 39.2, 54.3, 35.1, 43.8, 31.5,
            50.6, 36.4, 45.9, 38.1, 53.5, 33.7, 41.2, 37.8, 48.9, 34.2],
    'Heart_Rate': [68, 75, 62, 72, 65, 80, 58, 76, 70, 79,
                   64, 74, 60, 77, 67, 73, 61, 78, 69, 82,
                   63, 75, 66, 76, 59, 80, 71, 74, 65, 79],
    'RR_Mean': [14.2, 15.1, 13.8, 14.9, 14.5, 15.8, 13.2, 15.3, 14.6, 15.6,
                14.0, 15.2, 13.9, 15.4, 14.4, 15.0, 13.5, 15.7, 14.7, 16.0,
                13.7, 15.1, 14.3, 15.5, 13.6, 15.9, 14.8, 15.2, 14.1, 15.8],
    'BMI': [22.5, 24.3, 20.1, 26.8, 21.8, 28.5, 19.5, 27.2, 22.8, 29.1,
            21.2, 25.9, 20.8, 27.6, 23.1, 26.2, 20.5, 28.2, 22.4, 29.8,
            21.5, 25.4, 23.5, 27.0, 20.2, 28.8, 22.2, 26.5, 21.0, 28.1],
    'BMR': [1650, 1580, 1720, 1520, 1690, 1480, 1780, 1510, 1640, 1450,
            1710, 1550, 1750, 1490, 1670, 1530, 1770, 1460, 1660, 1420,
            1700, 1560, 1680, 1500, 1760, 1440, 1650, 1520, 1720, 1470],
    'Systolic_BP': [118, 125, 110, 132, 115, 138, 108, 128, 120, 135,
                    112, 130, 109, 126, 117, 131, 111, 136, 119, 140,
                    114, 127, 118, 129, 110, 137, 121, 132, 113, 134],
    'Diastolic_BP': [76, 82, 70, 85, 74, 88, 68, 83, 77, 87,
                     72, 84, 69, 82, 75, 85, 71, 86, 77, 89,
                     73, 81, 76, 84, 70, 87, 78, 86, 72, 85],
    'Temperature': [37.1, 37.2, 36.9, 37.3, 37.0, 37.4, 36.8, 37.2, 37.1, 37.5,
                    37.0, 37.3, 36.9, 37.2, 37.1, 37.4, 36.8, 37.5, 37.0, 37.6,
                    37.1, 37.2, 37.1, 37.3, 36.9, 37.5, 37.0, 37.4, 37.1, 37.4],
    'Age': [22, 25, 21, 28, 23, 30, 20, 26, 24, 29,
            22, 27, 21, 28, 25, 26, 23, 30, 24, 32,
            22, 27, 25, 29, 21, 31, 23, 28, 23, 30],
    'PCOS': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
             0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
             0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
print(f"✓ Dataset created successfully!")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nDataset Info:")
print(df.info())
print(f"\nSummary Statistics:")
print(df.describe())

# ========================================================================================================
# STEP 2: DATA EXPLORATION & EDA
# ========================================================================================================
print("\n[STEP 2: EXPLORATORY DATA ANALYSIS]")
print("-" * 100)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PCOS Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Class Distribution
class_counts = df['PCOS'].value_counts()
axes[0, 0].bar(['No PCOS (0)', 'PCOS (1)'], class_counts.values, color=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Class Distribution', fontweight='bold')
axes[0, 0].set_ylabel('Count')
for i, v in enumerate(class_counts.values):
    axes[0, 0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

print(f"✓ Class Distribution:")
print(f"  - PCOS=0: {class_counts[0]} ({class_counts[0]/len(df)*100:.1f}%)")
print(f"  - PCOS=1: {class_counts[1]} ({class_counts[1]/len(df)*100:.1f}%)")

# Correlation Heatmap
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0, 1], cbar_kws={'label': 'Correlation'})
axes[0, 1].set_title('Feature Correlation Heatmap', fontweight='bold')

# Age Distribution
df.boxplot(column='Age', by='PCOS', ax=axes[1, 0])
axes[1, 0].set_title('Age Distribution by PCOS Status', fontweight='bold')
axes[1, 0].set_xlabel('PCOS Status')
axes[1, 0].set_ylabel('Age (years)')

# BMI Distribution
df.boxplot(column='BMI', by='PCOS', ax=axes[1, 1])
axes[1, 1].set_title('BMI Distribution by PCOS Status', fontweight='bold')
axes[1, 1].set_xlabel('PCOS Status')
axes[1, 1].set_ylabel('BMI (kg/m²)')

plt.tight_layout()
plt.show()
print("✓ EDA visualization created!")

# ========================================================================================================
# STEP 3: FEATURE PREPARATION & NORMALIZATION
# ========================================================================================================
print("\n[STEP 3: FEATURE PREPARATION & NORMALIZATION]")
print("-" * 100)

feature_names = [col for col in df.columns if col != 'PCOS']
X = df[feature_names].values
y = df['PCOS'].values

print(f"✓ Feature matrix shape: {X.shape}")
print(f"✓ Target distribution: {np.bincount(y.astype(int))}")

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

print(f"✓ Features normalized using StandardScaler")
print(f"  - Mean (should be ~0): {X_normalized.mean(axis=0).round(6)}")
print(f"  - Std (should be ~1): {X_normalized.std(axis=0).round(3)}")

# ========================================================================================================
# STEP 4: SYNTHETIC DATA AUGMENTATION (SMOTE + GAUSSIAN NOISE)
# ========================================================================================================
print("\n[STEP 4: SYNTHETIC DATA AUGMENTATION]")
print("-" * 100)

print("\n⚠️  IMPORTANT: Synthetic data used for training robustness only!")
print("   Generated samples are mathematically interpolated between real samples.")
print("   Should NOT be used for clinical validation without real patient data.\n")

# Apply SMOTE
print("1. Applying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_smote, y_smote = smote.fit_resample(X_normalized, y)
print(f"   ✓ After SMOTE:")
print(f"     - New shape: {X_smote.shape}")
print(f"     - Class distribution: {np.bincount(y_smote.astype(int))}")

# Gaussian Noise Augmentation
print("\n2. Applying Gaussian Noise Augmentation...")
noise_scales = {
    'HRV': 2.0,
    'Heart_Rate': 3.0,
    'RR_Mean': 0.3,
    'BMI': 0.5,
    'BMR': 30.0,
    'Systolic_BP': 3.0,
    'Diastolic_BP': 2.0,
    'Temperature': 0.2,
    'Age': 0.0
}

X_augmented_list = [X_smote.copy()]
y_augmented_list = [y_smote.copy()]

for noise_factor in [0.5, 0.25]:
    X_noise = X_smote.copy()
    for feature_idx, feature_name in enumerate(feature_names):
        if feature_name in noise_scales:
            noise = np.random.normal(0, noise_scales[feature_name] * noise_factor, X_noise.shape[0])
            X_noise[:, feature_idx] += noise
    X_augmented_list.append(X_noise)
    y_augmented_list.append(y_smote.copy())

X_augmented = np.vstack(X_augmented_list)
y_augmented = np.hstack(y_augmented_list)

print(f"   ✓ After Augmentation:")
print(f"     - New shape: {X_augmented.shape}")
print(f"     - Class distribution: {np.bincount(y_augmented.astype(int))}")

# ========================================================================================================
# STEP 5: TRAIN-TEST SPLIT
# ========================================================================================================
print("\n[STEP 5: TRAIN-TEST SPLIT]")
print("-" * 100)

X_train, X_test, y_train, y_test = train_test_split(
    X_augmented, y_augmented, test_size=0.2, random_state=42, stratify=y_augmented
)

print(f"✓ Data split successfully:")
print(f"  - Training set: {X_train.shape[0]} samples")
print(f"  - Test set: {X_test.shape[0]} samples")
print(f"  - Train class distribution: {np.bincount(y_train.astype(int))}")
print(f"  - Test class distribution: {np.bincount(y_test.astype(int))}")

# ========================================================================================================
# STEP 6: MODEL TRAINING
# ========================================================================================================
print("\n[STEP 6: MODEL TRAINING]")
print("-" * 100)

print("\n1. Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
print("   ✓ Logistic Regression trained")

print("\n2. Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("   ✓ Random Forest trained")

print("\n3. Training XGBoost...")
xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
xgb.fit(X_train, y_train, verbose=False)
print("   ✓ XGBoost trained")

models = {
    'Logistic Regression': lr,
    'Random Forest': rf,
    'XGBoost': xgb
}

# ========================================================================================================
# STEP 7: MODEL EVALUATION & COMPARISON
# ========================================================================================================
print("\n[STEP 7: MODEL EVALUATION & COMPARISON]")
print("-" * 100)

results = []
predictions = {}
probabilities = {}

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    predictions[model_name] = y_pred
    probabilities[model_name] = y_pred_proba
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })
    
    print(f"  - Accuracy:  {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")
    print(f"  - ROC-AUC:   {roc_auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {cm[1,1]}, FP: {cm[0,1]}")
    print(f"    FN: {cm[1,0]}, TN: {cm[0,0]}")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No PCOS', 'PCOS']))

results_df = pd.DataFrame(results)
print("\n" + "=" * 100)
print("MODEL COMPARISON SUMMARY")
print("=" * 100)
print(results_df.to_string(index=False))

# ========================================================================================================
# STEP 8: VISUALIZATIONS
# ========================================================================================================
print("\n[STEP 8: VISUALIZATION OF RESULTS]")
print("-" * 100)

# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Confusion Matrices - Test Set', fontsize=14, fontweight='bold')

for idx, (model_name, model) in enumerate(models.items()):
    cm = confusion_matrix(y_test, predictions[model_name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
               xticklabels=['No PCOS', 'PCOS'],
               yticklabels=['No PCOS', 'PCOS'],
               cbar=False)
    axes[idx].set_title(model_name, fontweight='bold')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()
print("✓ Confusion matrices visualized")

# ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#3498db', '#e74c3c', '#2ecc71']
for idx, (model_name, model) in enumerate(models.items()):
    fpr, tpr, _ = roc_curve(y_test, probabilities[model_name])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("✓ ROC curves visualized")

# Feature Importance
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')

# Random Forest
rf_importance = rf.feature_importances_
rf_indices = np.argsort(rf_importance)[::-1]
axes[0].barh(np.array(feature_names)[rf_indices], rf_importance[rf_indices], color='#3498db')
axes[0].set_title('Random Forest Feature Importance', fontweight='bold')
axes[0].set_xlabel('Importance')

# XGBoost
xgb_importance = xgb.feature_importances_
xgb_indices = np.argsort(xgb_importance)[::-1]
axes[1].barh(np.array(feature_names)[xgb_indices], xgb_importance[xgb_indices], color='#e74c3c')
axes[1].set_title('XGBoost Feature Importance', fontweight='bold')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.show()
print("✓ Feature importance visualized")

# ========================================================================================================
# STEP 9: SAVE BEST MODEL
# ========================================================================================================
print("\n[STEP 9: BEST MODEL SELECTION]")
print("-" * 100)

best_idx = results_df['ROC-AUC'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_model = models[best_model_name]

print(f"\n✓ Best Model: {best_model_name}")
print(f"  - ROC-AUC Score: {results_df.loc[best_idx, 'ROC-AUC']:.4f}")
print(f"  - Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}")
print(f"  - F1-Score: {results_df.loc[best_idx, 'F1-Score']:.4f}")

# Save model (in Colab, this saves to the notebook environment)
best_model_name_file = best_model_name.replace(" ", "_")
try:
    joblib.dump(best_model, f'best_model_{best_model_name_file}.joblib')
    print(f"\n✓ Model saved as: best_model_{best_model_name_file}.joblib")
except:
    print(f"\n✓ Model ready for use (couldn't save file in this environment)")

# ========================================================================================================
# STEP 10: SUMMARY & PREDICTIONS ON NEW DATA
# ========================================================================================================
print("\n[STEP 10: FINAL SUMMARY]")
print("-" * 100)

print(f"\n✅ PIPELINE EXECUTION COMPLETE!")
print(f"\n📊 MODEL PERFORMANCES:")
print(results_df.to_string(index=False))

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"\n💾 HOW TO USE THE BEST MODEL:")
print(f"""
# For new patient prediction:
# new_patient = [[HRV, Heart_Rate, RR_Mean, BMI, BMR, Systolic_BP, Diastolic_BP, Temperature, Age]]
# Normalize: new_patient_scaled = scaler.transform(new_patient)
# Predict: prediction = best_model.predict(new_patient_scaled)
# Probability: probability = best_model.predict_proba(new_patient_scaled)
""")

print(f"\n⚠️  IMPORTANT LIMITATIONS:")
print(f"""
1. Synthetic data augmentation used for training robustness
2. Model trained on only 30 real samples + 60 synthetic samples
3. NOT suitable for clinical diagnosis without validation
4. Should be tested on real, unseen patient data
5. Requires clinical validation before any deployment
""")

print(f"\n✓ All stages completed successfully!")
print("=" * 100)

# Optional: Export results to CSV
results_df.to_csv('pcos_model_results.csv', index=False)
print("\n📁 Results saved to 'pcos_model_results.csv'")
