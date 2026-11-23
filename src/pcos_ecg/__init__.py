"""ECG feature pipeline and leakage-safe model comparison for PCOS screening research.

Research prototype only: not a medical device and not a diagnostic tool.
"""

import os

# XGBoost and PyTorch (used by SDV's CTGAN) each bundle an OpenMP runtime; on macOS the two
# crash with a segmentation fault once both run multi-threaded. The models here are small, so
# one thread costs little. Set before either library loads; a user's own value still wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.2.0"
