"""
Base Model Offline Training
Architecture: 4 -> 64 -> 64 -> 32 -> 4
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.dataset import FlowDatasetMinMax
from src.models import RegressionModel
from src.train_offline import train_offline, save_metrics_to_csv

DATA_FILE = str(PROJECT_ROOT / "data" / "ML_test_loader_original_data.csv")
OUTPUT_DIR = str(PROJECT_ROOT / "results" / "base_model_offline")
EPOCHS = 150
BATCH_SIZE = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset = FlowDatasetMinMax(DATA_FILE)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = RegressionModel().to(device)

metrics = train_offline(
    model=model,
    train_loader=train_loader,
    dataset=dataset,
    device=device,
    epochs=EPOCHS,
    model_name='base_model',
    output_dir=OUTPUT_DIR
)

save_metrics_to_csv(metrics, f"{OUTPUT_DIR}/base_model_metrics.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
epochs_range = range(1, len(metrics['loss']) + 1)

axes[0, 0].plot(epochs_range, metrics['loss'], 'b-', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontweight='bold')
axes[0, 0].set_ylabel('MSE Loss', fontweight='bold')
axes[0, 0].set_title('Training Loss (Normalized)', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(epochs_range, metrics['psnr'], 'g-', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontweight='bold')
axes[0, 1].set_ylabel('PSNR (dB)', fontweight='bold')
axes[0, 1].set_title('Peak Signal-to-Noise Ratio', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(epochs_range, metrics['ssim'], 'purple', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontweight='bold')
axes[1, 0].set_ylabel('SSIM', fontweight='bold')
axes[1, 0].set_title('Structural Similarity Index', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 1.05])

axes[1, 1].plot(epochs_range, metrics['relative_error'], 'r-', linewidth=2)
axes[1, 1].set_xlabel('Epoch', fontweight='bold')
axes[1, 1].set_ylabel('Relative Error (%)', fontweight='bold')
axes[1, 1].set_title('Reconstruction Error', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Base Model (64-64-32) - Offline Training',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/base_model_training_progress.png", dpi=300, bbox_inches='tight')
print(f"Training plot saved: {OUTPUT_DIR}/base_model_training_progress.png")
plt.close()

print("Base model offline training completed")
