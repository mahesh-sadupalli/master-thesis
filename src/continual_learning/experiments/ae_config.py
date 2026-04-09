"""
Configuration for Online Autoencoder Continual Learning Experiments

Provides model configs adapted for windowed input (60-dim instead of 1200-dim),
dataset paths, training hyperparameters, and offline reference metrics.
"""

import os
import sys
import torch

# ---------------------------------------------------------------------------
# Path resolution (same pattern as config.py)
# ---------------------------------------------------------------------------

if os.path.exists("/kaggle/input"):
    ENVIRONMENT = "kaggle"
    DATA_FILE = "/kaggle/input/ml-test-loader-original-data/ML_test_loader_original_data.csv"
    RESULTS_BASE = "/kaggle/working/results/autoencoder/linear_ae_online"
    SRC_DIR = "/kaggle/input/master-thesis-src/src"
elif os.path.exists("/content"):
    ENVIRONMENT = "colab"
    DATA_FILE = "/content/data/ML_test_loader_original_data.csv"
    RESULTS_BASE = "/content/results/autoencoder/linear_ae_online"
    SRC_DIR = "/content/src"
else:
    ENVIRONMENT = "local"
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    DATA_FILE = os.path.join(_root, "data", "ML_test_loader_original_data.csv")
    RESULTS_BASE = os.path.join(_root, "results", "autoencoder", "linear_ae_online")
    SRC_DIR = os.path.join(_root, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device():
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ---------------------------------------------------------------------------
# Online AE model configurations (input_dim = 15 timesteps * 4 vars = 60)
# ---------------------------------------------------------------------------

AE_ONLINE_MODEL_CONFIGS = {
    'base': {
        'encoder_hidden': [64, 32],
        'decoder_hidden': [32, 64],
        'latent_dim': 8,
        'dropout': 0.1,
    },
    'medium': {
        'encoder_hidden': [128, 64],
        'decoder_hidden': [64, 128],
        'latent_dim': 16,
        'dropout': 0.1,
    },
    'large': {
        'encoder_hidden': [256, 128, 64],
        'decoder_hidden': [64, 128, 256],
        'latent_dim': 32,
        'dropout': 0.1,
    },
}

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

LEARNING_RATE = 0.001
EPOCHS_PER_WINDOW = 100
NUM_WINDOWS = 20

# ---------------------------------------------------------------------------
# Strategy hyperparameters (matching INR CL experiments)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Offline reference metrics (from linear_ae_all_results.json)
# ---------------------------------------------------------------------------

AE_OFFLINE_REFERENCE = {
    "base":   {"psnr_db": 36.23, "ssim": 0.9697, "relative_error_pct": 2.74},
    "medium": {"psnr_db": 37.90, "ssim": 0.9719, "relative_error_pct": 2.26},
    "large":  {"psnr_db": 37.94, "ssim": 0.9749, "relative_error_pct": 2.25},
}
