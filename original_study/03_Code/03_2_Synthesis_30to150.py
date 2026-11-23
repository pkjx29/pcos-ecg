# Transcribed code-only copy of the Colab notebook "notusing_10_>150.ipynb"
# (team Google Drive folder "MP files unsorted").
# The notebook could not be downloaded because of the ">" in its file name, so its three
# cells are reproduced here verbatim. "# %%" marks a cell; VS Code and Jupyter can run it cell by cell.
# Despite the name, this is the notebook that wrote best_bayesian_150.csv, the dataset used by
# 03_3 (feature selection) and 03_4 (model comparison).

# %%
# ================================================================
# Direct Bayesian Network Synthesis: PCOS ECG-Centric Dataset
# ================================================================

# %capture
# Force specific protobuf version to resolve numbers-parser conflict
# %pip install -q -U "protobuf<5" sdv sdmetrics pandas numpy scipy scikit-learn matplotlib seaborn networkx umap-learn numbers-parser

import os
import sys
import json
import math
import random
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

try:
    from numbers_parser import Document
    NUMBERS_READER_AVAILABLE = True
except Exception as e:
    NUMBERS_READER_AVAILABLE = False

from scipy.stats import ks_2samp, wasserstein_distance, zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

import sdv
try:
    from sdv.metadata import Metadata
except Exception:
    from sdv.metadata import SingleTableMetadata
    Metadata = None

from sdmetrics.reports.single_table import QualityReport

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
SEEDS = [42, 123, 456, 789, 999]
N_SYNTHETIC = 150
TABLE_NAME = "pcos_ecg"
INPUT_FILE = "PCOS_ECG_Centric_14_Features_Corrected.numbers"
IDENTIFIER = "Subject_ID"
# Added 'Cycle Length' to categorical columns as it contains string descriptions
DECLARED_CATEGORICAL_COLUMNS = ["Diagnosis", "Target_Class", "Cycle Regularity", "Family History (PCOS/Diabetes)", "Cycle Length"]
PRIMARY_CLASS_COLUMN = "Target_Class"

OUTPUT_ROOT = Path("outputs")
DATASET_DIR = OUTPUT_ROOT / "datasets"
REPORT_DIR = OUTPUT_ROOT / "reports"
PLOT_DIR = OUTPUT_ROOT / "plots"
VALIDATION_DIR = OUTPUT_ROOT / "validation"

for directory in [OUTPUT_ROOT, DATASET_DIR, REPORT_DIR, PLOT_DIR, VALIDATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 300, "axes.titleweight": "bold"})

def reset_seeds(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def read_numbers_file(path):
    doc = Document(path)
    data = doc.sheets[0].tables[0].rows(values_only=True)
    return pd.DataFrame(data[1:], columns=data[0])

def clean_column_names(frame):
    frame = frame.copy()
    frame.columns = [str(c).strip().replace("\n", " ") for c in frame.columns]
    return frame

class BayesianNetworkSynthesizer:
    def __init__(self, categorical_columns, numeric_columns, class_column="Target_Class", id_column="Subject_ID", random_state=42):
        self.categorical_columns = list(categorical_columns)
        self.numeric_columns = list(numeric_columns)
        self.class_column = class_column
        self.id_column = id_column
        self.random_state = random_state
        self.fitted = False

    def fit(self, data):
        self.data = data.copy()
        self.rng = np.random.default_rng(self.random_state)
        self.class_counts = self.data[self.class_column].value_counts().to_dict()
        self.class_values = list(self.class_counts.keys())
        self.global_means = self.data[self.numeric_columns].mean()
        self.global_cov = self._stable_cov(self.data[self.numeric_columns])
        self.class_params = {cls: (self.data[self.data[self.class_column] == cls][self.numeric_columns].mean(),
                                   self._stable_cov(self.data[self.data[self.class_column] == cls][self.numeric_columns]))
                            for cls in self.class_values}
        self.cat_tables = {col: {cls: self.data[self.data[self.class_column] == cls][col].value_counts(normalize=True)
                                 for cls in self.class_values} for col in self.categorical_columns if col != self.class_column}
        self.fitted = True
        return self

    def _stable_cov(self, frame):
        if len(frame) < 2: return np.eye(len(self.numeric_columns)) * 1e-4
        cov = np.cov(frame.to_numpy(dtype=float), rowvar=False)
        return 0.7 * cov + 0.3 * np.diag(np.diag(cov)) + np.eye(cov.shape[0]) * 1e-6

    def sample(self, num_rows, seed=None):
        rng = np.random.default_rng(seed or self.random_state)
        classes = rng.choice(self.class_values, size=num_rows)
        rows = []
        for i, cls in enumerate(classes):
            means, cov = self.class_params[cls]
            num_vals = rng.multivariate_normal(means, cov)
            record = dict(zip(self.numeric_columns, num_vals))
            for col, tables in self.cat_tables.items():
                record[col] = rng.choice(tables[cls].index, p=tables[cls].values)
            record[self.class_column] = cls
            record[self.id_column] = f"SYN_{i:04d}"
            rows.append(record)
        return pd.DataFrame(rows)[self.data.columns]

try:
    input_path = Path("/content") / INPUT_FILE
    original = clean_column_names(read_numbers_file(input_path))

    num_cols = [c for c in original.columns if c not in DECLARED_CATEGORICAL_COLUMNS + [IDENTIFIER]]
    for c in num_cols:
        original[c] = pd.to_numeric(original[c], errors="coerce").fillna(original[c].median() if not original[c].isna().all() else 0)

    # Ensure categorical columns are strings to avoid mixed-type issues
    for c in DECLARED_CATEGORICAL_COLUMNS:
        original[c] = original[c].astype(str)

    synth = BayesianNetworkSynthesizer(DECLARED_CATEGORICAL_COLUMNS, num_cols, PRIMARY_CLASS_COLUMN, IDENTIFIER)
    synth.fit(original)
    best_synthetic = synth.sample(N_SYNTHETIC)

    best_synthetic.to_csv(DATASET_DIR / "best_bayesian_150.csv", index=False)
    print("Synthesis Complete. Samples generated: 150")
    display(best_synthetic.head())
except Exception as e:
    print(f"Error: {e}")

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sdmetrics.reports.single_table import QualityReport

# 1. Prepare Metadata for SDMetrics
try:
    columns_metadata = {col: {'sdtype': 'numerical'} for col in num_cols}
    for col in DECLARED_CATEGORICAL_COLUMNS:
        columns_metadata[col] = {'sdtype': 'categorical'}
    columns_metadata[IDENTIFIER] = {'sdtype': 'id'}

    metadata_dict = {
        'columns': columns_metadata,
        'primary_key': IDENTIFIER
    }

    # 2. Run SDMetrics Quality Report
    report = QualityReport()
    report.generate(original, best_synthetic, metadata_dict)
    print("\n--- SDMetrics Quality Report Summary ---")
    display(report.get_score())
    display(report.get_details(property_name='Column Shapes'))
except Exception as e:
    print(f"SDMetrics Report Error: {e}")

# 3. Visual Validation: PCA
scaler = StandardScaler()
real_scaled = scaler.fit_transform(original[num_cols])
synth_scaled = scaler.transform(best_synthetic[num_cols])

pca = PCA(n_components=2)
real_pca = pca.fit_transform(real_scaled)
synth_pca = pca.transform(synth_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.3, label='Synthetic', c='orange')
plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.8, label='Real', c='blue', edgecolors='white')
plt.title('PCA: Real vs Synthetic Data Distribution')
plt.legend()
plt.savefig(PLOT_DIR / 'pca_comparison.png')
plt.show()

# 4. Correlation Heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
real_corr = original[num_cols].corr()
synth_corr = best_synthetic[num_cols].corr()

sns.heatmap(real_corr, annot=False, cmap='coolwarm', ax=ax1)
ax1.set_title('Real Data Correlations')

sns.heatmap(synth_corr, annot=False, cmap='coolwarm', ax=ax2)
ax2.set_title('Synthetic Data Correlations')

plt.tight_layout()
plt.savefig(PLOT_DIR / 'correlation_comparison.png')
plt.show()

# %%
from google.colab import files
import zipfile

# Function to zip and download all outputs
def download_outputs():
    zip_path = "pcos_synthesis_results.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files_list in os.walk(OUTPUT_ROOT):
            for file in files_list:
                zipf.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file), OUTPUT_ROOT))

    print(f"Results zipped to {zip_path}. Triggering download...")
    files.download(zip_path)

download_outputs()
