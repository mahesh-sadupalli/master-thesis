"""
Medium Model Online Training

Trains medium compressor (4→96→96→48→4) using temporal window approach
to simulate streaming in-situ compression. Processes data in 20 windows with 100
epochs per window for balanced performance between base and large models.
"""

import torch
import matplotlib.pyplot as plt
import pandas as pd
from unified_training_utils import SpatioTemporalDataset, MediumCompressor
from online_training_utils import train_online

DATA_FILE = "../data/ML_test_loader_original_data.csv"
OUTPUT_DIR = "../results/medium_model_online"
EPOCHS_PER_WINDOW = 100
NUM_WINDOWS = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset = SpatioTemporalDataset(DATA_FILE)
model = MediumCompressor().to(device)

metrics = train_online(
    model=model,
    dataset=dataset,
    device=device,
    epochs_per_window=EPOCHS_PER_WINDOW,
    model_name='medium_model_online',
    output_dir=OUTPUT_DIR,
    num_windows=NUM_WINDOWS
)

# Save metrics to CSV
df = pd.DataFrame(metrics)
df.to_csv(f"{OUTPUT_DIR}/training_metrics.csv", index=False)
print(f"Metrics saved: {OUTPUT_DIR}/training_metrics.csv")

# Create training progress plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(metrics['window'], metrics['loss'], 'b-o', linewidth=2, markersize=4)
axes[0, 0].set_xlabel('Window')
axes[0, 0].set_ylabel('MSE Loss')
axes[0, 0].set_title('Loss per Window', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(metrics['window'], metrics['psnr'], 'g-o', linewidth=2, markersize=4)
axes[0, 1].set_xlabel('Window')
axes[0, 1].set_ylabel('PSNR (dB)')
axes[0, 1].set_title('Peak Signal-to-Noise Ratio', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(metrics['window'], metrics['ssim'], 'purple', linewidth=2, marker='o', markersize=4)
axes[1, 0].set_xlabel('Window')
axes[1, 0].set_ylabel('SSIM')
axes[1, 0].set_title('Structural Similarity Index', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 1.05])

axes[1, 1].plot(metrics['window'], metrics['relative_error'], 'r-o', linewidth=2, markersize=4)
axes[1, 1].set_xlabel('Window')
axes[1, 1].set_ylabel('Relative Error (%)')
axes[1, 1].set_title('Reconstruction Error', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f'Medium Model (96-96-48) - Online Training - {NUM_WINDOWS} Windows',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_progress.png", dpi=300, bbox_inches='tight')
print(f"Training plot saved: {OUTPUT_DIR}/training_progress.png")
plt.close()

print("Medium model online training completed")
