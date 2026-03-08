"""
Shared Configuration for Continual Learning Experiments

Provides dataset paths, model definitions, training hyperparameters,
and device configuration. Supports local, Kaggle, and Colab environments.
"""

import os
import sys
import torch

# ---------------------------------------------------------------------------
# Path resolution: works on local machine, Kaggle, and Colab
# ---------------------------------------------------------------------------

# Detect environment
if os.path.exists("/kaggle/input"):
    ENVIRONMENT = "kaggle"
    DATA_FILE = "/kaggle/input/ml-test-loader-original-data/ML_test_loader_original_data.csv"
    RESULTS_BASE = "/kaggle/working/results/continual_learning"
    SRC_DIR = "/kaggle/input/master-thesis-src/src"
elif os.path.exists("/content"):
    ENVIRONMENT = "colab"
    DATA_FILE = "/content/data/ML_test_loader_original_data.csv"
    RESULTS_BASE = "/content/results/continual_learning"
    SRC_DIR = "/content/src"
else:
    ENVIRONMENT = "local"
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    DATA_FILE = os.path.join(_root, "data", "ML_test_loader_original_data.csv")
    RESULTS_BASE = os.path.join(_root, "results", "continual_learning")
    SRC_DIR = os.path.join(_root, "src")

# Ensure src is importable
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# Device selection (CUDA > MPS > CPU)
# ---------------------------------------------------------------------------

def get_device():
    """Select the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "base":   "BaseCompressor",    # 6,692 params,  26.8 KB
    "medium": "MediumCompressor",  # 14,644 params, 58.6 KB
    "large":  "LargeCompressor",   # 25,668 params, 102.7 KB
}

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

LEARNING_RATE = 0.001
EPOCHS_PER_WINDOW = 100
NUM_WINDOWS = 20
BATCH_SIZE = 512  # Used only for offline training reference

# ---------------------------------------------------------------------------
# Strategy default hyperparameters
# ---------------------------------------------------------------------------

ER_DEFAULTS = {
    "buffer_size": 10000,
    "replay_weight": 0.5,
    "replay_batch_size": 5000,
}

ER_SCALED_DEFAULTS = {
    "buffer_size": 50000,
    "replay_weight": 0.7,
    "replay_batch_size": 10000,
}

ER_AGGRESSIVE_DEFAULTS = {
    "buffer_size": 100000,
    "replay_weight": 1.0,
    "replay_batch_size": 20000,
}

EWC_DEFAULTS = {
    "ewc_lambda": 1000.0,
    "fisher_samples": 50000,
}

LWF_DEFAULTS = {
    "distill_weight": 0.5,
}

DERPP_DEFAULTS = {
    "buffer_size": 10000,
    "replay_weight": 0.5,
    "distill_weight": 0.5,
    "replay_batch_size": 5000,
}

COMBINED_DEFAULTS = {
    "buffer_size": 10000,
    "replay_weight": 0.3,
    "distill_weight": 0.3,
    "ewc_lambda": 500.0,
    "replay_batch_size": 5000,
}

# ---------------------------------------------------------------------------
# Offline reference metrics (from previous experiments)
# ---------------------------------------------------------------------------

OFFLINE_REFERENCE = {
    "base":   {"psnr_db": 32.15, "ssim": 0.9550, "relative_error_pct": 4.41},
    "medium": {"psnr_db": 33.58, "ssim": 0.9583, "relative_error_pct": 3.74},
    "large":  {"psnr_db": 35.99, "ssim": 0.9856, "relative_error_pct": 2.83},
}
