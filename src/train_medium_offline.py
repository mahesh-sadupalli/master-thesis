"""
Medium Model Offline Training
Architecture: 4 → 96 → 96 → 48 → 4
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from unified_training_utils import (
    SpatioTemporalDataset,
    MediumCompressor,
    train_model,
    export_metrics_csv
)

DATA_FILE = "../data/ML_test_loader_original_data.csv"
OUTPUT_DIR = "../results/medium_model_offline"
EPOCHS = 150
BATCH_SIZE = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset = SpatioTemporalDataset(DATA_FILE)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = MediumCompressor().to(device)

metrics = train_model(
    model=model,
    train_loader=train_loader,
    dataset=dataset,
    device=device,
    num_epochs=EPOCHS,
    model_name='medium_model_offline',
    output_dir=OUTPUT_DIR
)

export_metrics_csv(metrics, f"{OUTPUT_DIR}/medium_model_metrics.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
epochs_range = range(1, len(metrics['loss']) + 1)

axes[0, 0].plot(epochs_range, metrics['loss'], 'b-', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MSE Loss')
axes[0, 0].set_title('Training Loss (Normalized)', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(epochs_range, metrics['psnr'], 'g-', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('PSNR (dB)')
axes[0, 1].set_title('Peak Signal-to-Noise Ratio', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(epochs_range, metrics['ssim'], 'purple', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('SSIM')
axes[1, 0].set_title('Structural Similarity Index', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 1.05])

axes[1, 1].plot(epochs_range, metrics['relative_error'], 'r-', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Relative Error (%)')
axes[1, 1].set_title('Reconstruction Error', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Medium Model (96-96-48) - Offline Training',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/medium_model_training_progress.png", dpi=300, bbox_inches='tight')
print(f"Training plot saved: {OUTPUT_DIR}/medium_model_training_progress.png")
plt.close()

print("Medium model offline training completed")
