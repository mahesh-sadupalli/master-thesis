"""
Advanced Model Online Training
Architecture: 4 -> 128 -> 128 -> 64 -> 4
Simulates streaming in-situ training with temporal windows
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import matplotlib.pyplot as plt
import pandas as pd
from src.dataset import FlowDatasetMinMax
from src.models import AdvancedRegressionModel
from src.train_online import train_online

DATA_FILE = str(PROJECT_ROOT / "data" / "ML_test_loader_original_data.csv")
OUTPUT_DIR = str(PROJECT_ROOT / "results" / "large_model_online")
EPOCHS_PER_WINDOW = 100
NUM_WINDOWS = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset = FlowDatasetMinMax(DATA_FILE)
model = AdvancedRegressionModel().to(device)

metrics = train_online(
    model=model,
    dataset=dataset,
    device=device,
    epochs_per_window=EPOCHS_PER_WINDOW,
    model_name='advanced_model',
    output_dir=OUTPUT_DIR,
    num_windows=NUM_WINDOWS
)

df = pd.DataFrame(metrics)
df.to_csv(f"{OUTPUT_DIR}/advanced_model_online_metrics.csv", index=False)
print(f"Metrics saved: {OUTPUT_DIR}/advanced_model_online_metrics.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(metrics['window'], metrics['loss'], 'b-o', linewidth=2, markersize=4)
axes[0, 0].set_xlabel('Window', fontweight='bold')
axes[0, 0].set_ylabel('MSE Loss', fontweight='bold')
axes[0, 0].set_title('Loss per Window (Normalized)', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(metrics['window'], metrics['psnr'], 'g-o', linewidth=2, markersize=4)
axes[0, 1].set_xlabel('Window', fontweight='bold')
axes[0, 1].set_ylabel('PSNR (dB)', fontweight='bold')
axes[0, 1].set_title('Peak Signal-to-Noise Ratio', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(metrics['window'], metrics['ssim'], 'purple', linewidth=2, marker='o', markersize=4)
axes[1, 0].set_xlabel('Window', fontweight='bold')
axes[1, 0].set_ylabel('SSIM', fontweight='bold')
axes[1, 0].set_title('Structural Similarity Index', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 1.05])

axes[1, 1].plot(metrics['window'], metrics['relative_error'], 'r-o', linewidth=2, markersize=4)
axes[1, 1].set_xlabel('Window', fontweight='bold')
axes[1, 1].set_ylabel('Relative Error (%)', fontweight='bold')
axes[1, 1].set_title('Reconstruction Error', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f'Advanced Model (128-128-64) - Online Training - {NUM_WINDOWS} Windows',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/advanced_model_online_progress.png", dpi=300, bbox_inches='tight')
print(f"Training plot saved: {OUTPUT_DIR}/advanced_model_online_progress.png")
plt.close()

print("Advanced model online training completed")
