"""
===================================================================================================
PCOS RISK SCREENING - MACHINE LEARNING PIPELINE
===================================================================================================
Project: Early PCOS Risk Screening using Physiological Parameters
Description: A beginner-friendly ML pipeline for PCOS risk classification
Author: Biomedical Engineering Student
Date: 2025

Key Features:
- Data loading and exploratory analysis
- Synthetic data augmentation (SMOTE + Gaussian noise)
- Feature normalization
- Multi-model comparison (Logistic Regression, Random Forest, XGBoost)
- Comprehensive evaluation and visualization

Dataset Features: HRV, Heart_Rate, RR_Mean, BMI, BMR, Systolic_BP, Diastolic_BP, Temperature, Age
Target: PCOS (0 = No PCOS, 1 = PCOS positive)
===================================================================================================
"""

# ===================================================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ===================================================================================================
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

# Configure visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 100)
print("PCOS RISK SCREENING - ML PIPELINE")
print("=" * 100)


# ===================================================================================================
# SECTION 2: DATA LOADING AND INITIAL EXPLORATION
# ===================================================================================================
def load_and_explore_data(filepath):
    """
    Load dataset and perform initial exploratory analysis.
    
    Parameters:
    - filepath: Path to CSV file containing PCOS data
    
    Returns:
    - df: Loaded pandas DataFrame
    """
    print("\n[STAGE 1: DATA LOADING AND EXPLORATION]")
    print("-" * 100)
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Display first few rows
    print("\nFirst 5 rows of dataset:")
    print(df.head())
    
    # Basic statistics
    print("\nDataset Info:")
    print(df.info())
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
        print("⚠ Missing values detected! Handling...")
        df = df.dropna()  # Remove rows with missing values
        print(f"✓ Cleaned dataset shape: {df.shape}")
    else:
        print("✓ No missing values detected!")
    
    return df


# ===================================================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# ===================================================================================================
def perform_eda(df):
    """
    Perform comprehensive exploratory data analysis.
    Creates visualizations for:
    - Class distribution
    - Correlation heatmap
    - Feature distributions
    """
    print("\n[STAGE 2: EXPLORATORY DATA ANALYSIS]")
    print("-" * 100)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PCOS Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Class Distribution
    class_counts = df['PCOS'].value_counts()
    axes[0, 0].bar(['No PCOS (0)', 'PCOS (1)'], class_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title('Class Distribution (Before Augmentation)', fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(class_counts.values):
        axes[0, 0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    
    print(f"✓ Class Distribution:")
    print(f"  - PCOS=0 (Negative): {class_counts[0]} samples ({class_counts[0]/len(df)*100:.1f}%)")
    print(f"  - PCOS=1 (Positive): {class_counts[1]} samples ({class_counts[1]/len(df)*100:.1f}%)")
    
    # 2. Feature Correlation Heatmap
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0, 1], 
                cbar_kws={'label': 'Correlation'})
    axes[0, 1].set_title('Feature Correlation Heatmap', fontweight='bold')
    
    # 3. Age Distribution by PCOS Status
    df.boxplot(column='Age', by='PCOS', ax=axes[1, 0])
    axes[1, 0].set_title('Age Distribution by PCOS Status', fontweight='bold')
    axes[1, 0].set_xlabel('PCOS Status')
    axes[1, 0].set_ylabel('Age (years)')
    
    # 4. BMI Distribution by PCOS Status
    df.boxplot(column='BMI', by='PCOS', ax=axes[1, 1])
    axes[1, 1].set_title('BMI Distribution by PCOS Status', fontweight='bold')
    axes[1, 1].set_xlabel('PCOS Status')
    axes[1, 1].set_ylabel('BMI (kg/m²)')
    
    plt.tight_layout()
    plt.savefig('output/01_eda_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ EDA visualization saved: output/01_eda_analysis.png")
    plt.show()


# ===================================================================================================
# SECTION 4: FEATURE PREPARATION AND NORMALIZATION
# ===================================================================================================
def prepare_features(df):
    """
    Separate features (X) and target (y), and normalize features.
    
    Parameters:
    - df: Input DataFrame
    
    Returns:
    - X_normalized: Normalized features
    - y: Target variable
    - scaler: Fitted StandardScaler object (for inverse transforms if needed)
    - feature_names: List of feature names
    """
    print("\n[STAGE 3: FEATURE PREPARATION AND NORMALIZATION]")
    print("-" * 100)
    
    # Separate features and target
    feature_names = [col for col in df.columns if col != 'PCOS']
    X = df[feature_names].values
    y = df['PCOS'].values
    
    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Target distribution: {np.bincount(y.astype(int))}")
    
    # Normalize features using StandardScaler
    # IMPORTANT: Normalization centers data around 0 with standard deviation of 1
    # This helps machine learning algorithms perform better
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    print(f"✓ Features normalized using StandardScaler")
    print(f"  - Mean (should be ~0): {X_normalized.mean(axis=0).round(6)}")
    print(f"  - Std (should be ~1): {X_normalized.std(axis=0).round(3)}")
    
    return X_normalized, y, scaler, feature_names


# ===================================================================================================
# SECTION 5: SYNTHETIC DATA AUGMENTATION
# ===================================================================================================
def augment_data(X, y, feature_names):
    """
    Apply synthetic data augmentation using:
    1. SMOTE (Synthetic Minority Over-sampling Technique)
    2. Gaussian noise injection
    
    IMPORTANT NOTE ON SYNTHETIC DATA:
    ================================
    Synthetic data augmentation is necessary for small medical datasets to prevent overfitting.
    However, generated data should be validated to ensure physiological realism.
    
    LIMITATIONS:
    - Generated values may not capture complex patient interactions
    - Should NEVER replace clinical judgment
    - Results on synthetic data should be validated on real holdout data
    
    Parameters:
    - X: Normalized feature matrix
    - y: Target vector
    - feature_names: List of feature names
    
    Returns:
    - X_augmented: Augmented feature matrix
    - y_augmented: Augmented target vector
    """
    print("\n[STAGE 4: SYNTHETIC DATA AUGMENTATION]")
    print("-" * 100)
    
    print("\n⚠️  IMPORTANT: This pipeline uses SMOTE for class balancing.")
    print("   Generated synthetic samples are interpolations between existing samples.")
    print("   While useful for training, predictions on real data must be validated clinically.")
    
    # Step 1: Apply SMOTE
    print("\n1. Applying SMOTE (Synthetic Minority Over-sampling Technique)...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_smote, y_smote = smote.fit_resample(X, y)
    
    print(f"   ✓ After SMOTE:")
    print(f"     - New shape: {X_smote.shape}")
    print(f"     - Class distribution: {np.bincount(y_smote.astype(int))}")
    
    # Step 2: Optional Gaussian Noise Augmentation
    print("\n2. Applying Gaussian Noise Augmentation...")
    print("   Creating synthetic variations with realistic physiological noise...")
    
    # Define physiologically reasonable noise levels for each feature
    # These values are based on typical measurement variations
    noise_scales = {
        'HRV': 2.0,              # beats/ms - small variation
        'Heart_Rate': 3.0,       # bpm
        'RR_Mean': 0.3,          # breaths/min
        'BMI': 0.5,              # kg/m²
        'BMR': 30.0,             # kcal/day
        'Systolic_BP': 3.0,      # mmHg
        'Diastolic_BP': 2.0,     # mmHg
        'Temperature': 0.2,      # °C
        'Age': 0.0               # Age doesn't change (no noise for age)
    }
    
    X_augmented_list = [X_smote.copy()]
    y_augmented_list = [y_smote.copy()]
    
    # Create 2 additional augmented versions per original sample
    for noise_factor in [0.5, 0.25]:  # Create 2 augmented copies with different noise levels
        X_noise = X_smote.copy()
        
        for feature_idx, feature_name in enumerate(feature_names):
            if feature_name in noise_scales:
                noise = np.random.normal(0, noise_scales[feature_name] * noise_factor, X_noise.shape[0])
                X_noise[:, feature_idx] += noise
        
        X_augmented_list.append(X_noise)
        y_augmented_list.append(y_smote.copy())
    
    # Combine all augmented data
    X_augmented = np.vstack(X_augmented_list)
    y_augmented = np.hstack(y_augmented_list)
    
    print(f"   ✓ After Gaussian Noise Augmentation:")
    print(f"     - New shape: {X_augmented.shape}")
    print(f"     - Class distribution: {np.bincount(y_augmented.astype(int))}")
    
    # Verify data stays within reasonable ranges
    print("\n3. Verifying Physiological Realism...")
    # (In a real scenario, you'd compare against known physiological ranges)
    print(f"   ✓ Feature ranges after augmentation:")
    for idx, feature_name in enumerate(feature_names):
        min_val = X_augmented[:, idx].min()
        max_val = X_augmented[:, idx].max()
        mean_val = X_augmented[:, idx].mean()
        print(f"     - {feature_name:15}: [{min_val:8.2f}, {max_val:8.2f}] (mean: {mean_val:7.2f})")
    
    return X_augmented, y_augmented


# ===================================================================================================
# SECTION 6: TRAIN-TEST SPLIT
# ===================================================================================================
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split augmented data into training and test sets.
    
    Parameters:
    - X: Feature matrix (augmented)
    - y: Target vector (augmented)
    - test_size: Proportion of data to use for testing
    - random_state: Random seed for reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test
    """
    print("\n[STAGE 5: TRAIN-TEST SPLIT]")
    print("-" * 100)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Data split successfully:")
    print(f"  - Training set: {X_train.shape[0]} samples")
    print(f"  - Test set: {X_test.shape[0]} samples")
    print(f"  - Train class distribution: {np.bincount(y_train.astype(int))}")
    print(f"  - Test class distribution: {np.bincount(y_test.astype(int))}")
    
    return X_train, X_test, y_train, y_test


# ===================================================================================================
# SECTION 7: MODEL TRAINING
# ===================================================================================================
def train_models(X_train, y_train):
    """
    Train three different machine learning models.
    
    Models:
    1. Logistic Regression: Fast, interpretable linear model
    2. Random Forest: Ensemble method, handles non-linearity well
    3. XGBoost: Gradient boosting, often best performance
    
    Parameters:
    - X_train: Training features
    - y_train: Training target
    
    Returns:
    - Dictionary of trained models
    """
    print("\n[STAGE 6: MODEL TRAINING]")
    print("-" * 100)
    
    models = {}
    
    # 1. Logistic Regression
    print("\n1. Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    print("   ✓ Logistic Regression trained")
    print(f"     - Coefficients: {lr.coef_[0].round(4)}")
    
    # 2. Random Forest
    print("\n2. Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    print("   ✓ Random Forest trained")
    print(f"     - Number of trees: 100")
    
    # 3. XGBoost
    print("\n3. Training XGBoost Classifier...")
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train, verbose=False)
    models['XGBoost'] = xgb
    print("   ✓ XGBoost trained")
    
    return models


# ===================================================================================================
# SECTION 8: MODEL EVALUATION AND COMPARISON
# ===================================================================================================
def evaluate_models(models, X_test, y_test, feature_names):
    """
    Evaluate all models on test set and compare performance.
    
    Metrics:
    - Accuracy: Overall correctness
    - Precision: True positives / (True positives + False positives)
    - Recall: True positives / (True positives + False negatives)
    - F1-Score: Harmonic mean of precision and recall
    - ROC-AUC: Area under ROC curve (0.5 = random, 1.0 = perfect)
    
    Parameters:
    - models: Dictionary of trained models
    - X_test: Test features
    - y_test: Test target
    - feature_names: List of feature names
    
    Returns:
    - results_df: DataFrame with evaluation metrics
    """
    print("\n[STAGE 7: MODEL EVALUATION AND COMPARISON]")
    print("-" * 100)
    
    results = []
    predictions = {}
    probabilities = {}
    
    # Evaluate each model
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Store predictions
        predictions[model_name] = y_pred
        probabilities[model_name] = y_pred_proba
        
        # Calculate metrics
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
        
        # Print detailed results
        print(f"  - Accuracy:  {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall:    {recall:.4f}")
        print(f"  - F1-Score:  {f1:.4f}")
        print(f"  - ROC-AUC:   {roc_auc:.4f}")
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"    True Negatives:  {cm[0,0]}")
        print(f"    False Positives: {cm[0,1]}")
        print(f"    False Negatives: {cm[1,0]}")
        print(f"    True Positives:  {cm[1,1]}")
        
        # Classification report
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No PCOS', 'PCOS']))
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 100)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 100)
    print(results_df.to_string(index=False))
    
    return results_df, predictions, probabilities


# ===================================================================================================
# SECTION 9: VISUALIZATION OF RESULTS
# ===================================================================================================
def visualize_results(models, predictions, probabilities, y_test, feature_names):
    """
    Create comprehensive visualizations:
    - Confusion matrices for each model
    - ROC curves for each model
    - Feature importance plots
    
    Parameters:
    - models: Dictionary of trained models
    - predictions: Dictionary of predictions
    - probabilities: Dictionary of prediction probabilities
    - y_test: True test labels
    - feature_names: List of feature names
    """
    print("\n[STAGE 8: VISUALIZATION OF RESULTS]")
    print("-" * 100)
    
    # 1. Confusion Matrices
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
    plt.savefig('output/02_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrices saved: output/02_confusion_matrices.png")
    plt.show()
    
    # 2. ROC Curves
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
    
    plt.savefig('output/03_roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ ROC curves saved: output/03_roc_curves.png")
    plt.show()
    
    # 3. Feature Importance
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')
    
    # Random Forest Feature Importance
    rf_model = models['Random Forest']
    rf_importance = rf_model.feature_importances_
    rf_indices = np.argsort(rf_importance)[::-1]
    
    axes[0].barh(np.array(feature_names)[rf_indices], rf_importance[rf_indices], color='#3498db')
    axes[0].set_title('Random Forest Feature Importance', fontweight='bold')
    axes[0].set_xlabel('Importance')
    
    # XGBoost Feature Importance
    xgb_model = models['XGBoost']
    xgb_importance = xgb_model.feature_importances_
    xgb_indices = np.argsort(xgb_importance)[::-1]
    
    axes[1].barh(np.array(feature_names)[xgb_indices], xgb_importance[xgb_indices], color='#e74c3c')
    axes[1].set_title('XGBoost Feature Importance', fontweight='bold')
    axes[1].set_xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('output/04_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Feature importance plot saved: output/04_feature_importance.png")
    plt.show()


# ===================================================================================================
# SECTION 10: MODEL PERSISTENCE
# ===================================================================================================
def save_best_model(models, results_df):
    """
    Save the best performing model using joblib.
    
    The best model is selected based on ROC-AUC score.
    
    Parameters:
    - models: Dictionary of trained models
    - results_df: DataFrame with evaluation results
    """
    print("\n[STAGE 9: MODEL PERSISTENCE]")
    print("-" * 100)
    
    # Find best model based on ROC-AUC
    best_idx = results_df['ROC-AUC'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_model = models[best_model_name]
    
    # Save model
    model_path = f'models/best_model_{best_model_name.replace(" ", "_")}.joblib'
    joblib.dump(best_model, model_path)
    
    print(f"✓ Best model saved: {model_path}")
    print(f"  - Model: {best_model_name}")
    print(f"  - ROC-AUC Score: {results_df.loc[best_idx, 'ROC-AUC']:.4f}")
    
    # Also save the scaler (will need to add this)
    # scaler_path = 'models/scaler.joblib'
    # joblib.dump(scaler, scaler_path)
    # print(f"✓ Scaler saved: {scaler_path}")
    
    return best_model_name, best_model


# ===================================================================================================
# SECTION 11: MAIN PIPELINE EXECUTION
# ===================================================================================================
def main():
    """
    Main pipeline execution function.
    Orchestrates all stages of the ML workflow.
    """
    
    # Stage 1: Load and explore data
    df = load_and_explore_data('data/pcos_data_sample.csv')
    
    # Stage 2: Perform EDA
    perform_eda(df)
    
    # Stage 3: Prepare features
    X_normalized, y, scaler, feature_names = prepare_features(df)
    
    # Stage 4: Augment data
    X_augmented, y_augmented = augment_data(X_normalized, y, feature_names)
    
    # Stage 5: Split data
    X_train, X_test, y_train, y_test = split_data(X_augmented, y_augmented)
    
    # Stage 6: Train models
    models = train_models(X_train, y_train)
    
    # Stage 7: Evaluate models
    results_df, predictions, probabilities = evaluate_models(models, X_test, y_test, feature_names)
    
    # Stage 8: Visualize results
    visualize_results(models, predictions, probabilities, y_test, feature_names)
    
    # Stage 9: Save best model
    best_model_name, best_model = save_best_model(models, results_df)
    
    # Final summary
    print("\n" + "=" * 100)
    print("PIPELINE EXECUTION COMPLETE!")
    print("=" * 100)
    print(f"\n✓ All results saved to 'output/' directory")
    print(f"✓ Best model saved to 'models/' directory")
    print(f"✓ Best performing model: {best_model_name}")
    
    return results_df, models, best_model


# ===================================================================================================
# ENTRY POINT
# ===================================================================================================
if __name__ == "__main__":
    results_df, models, best_model = main()
