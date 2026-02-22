"""
Base Model Online Visualization

Generates spatio-temporal visualizations and error analysis for trained base model
from online streaming training with normalized colorbars in [0, 1] range.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from unified_training_utils import SpatioTemporalDataset, BaseCompressor, compute_psnr_ssim

DATA_FILE = "../data/ML_test_loader_original_data.csv"
MODEL_DIR = "../results/base_model_online"
OUTPUT_DIR = "../results/base_model_online"
TIMESTEP = 0.0396

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Base model online visualization")
print(f"Device: {device}")

dataset = SpatioTemporalDataset(DATA_FILE)

model = BaseCompressor().to(device)
model.load_state_dict(torch.load(f"{MODEL_DIR}/base_model_online_final.pth", map_location=device))
model.eval()

print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

print(f"Generating predictions for {len(dataset):,} points")
all_inputs = torch.from_numpy(dataset.inputs).to(device)
with torch.no_grad():
    all_predictions = model(all_inputs).cpu()

all_targets = torch.from_numpy(dataset.targets)

psnr, ssim = compute_psnr_ssim(all_predictions, all_targets, device)
print(f"\nEvaluation metrics:")
print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim:.4f}")

# Use normalized data for visualization (already in [0, 1] range)
predictions_norm = all_predictions.numpy()
targets_norm = all_targets.numpy()
coords_denorm = dataset.denormalize_inputs(dataset.inputs)

timestep_mask = np.abs(coords_denorm[:, 3] - TIMESTEP) < 1e-6
x = coords_denorm[timestep_mask, 0]
y = coords_denorm[timestep_mask, 1]
pred_t = predictions_norm[timestep_mask]
target_t = targets_norm[timestep_mask]
errors_t = np.abs(target_t - pred_t)

print(f"Visualizing {len(x):,} points at timestep {TIMESTEP}")

feature_names = ['Vx', 'Vy', 'Pressure', 'TKE']
feature_indices = [0, 1, 2, 3]

fig, axes = plt.subplots(4, 3, figsize=(18, 20))

for row, (idx, name) in enumerate(zip(feature_indices, feature_names)):
    original = target_t[:, idx]
    predicted = pred_t[:, idx]
    error = errors_t[:, idx]

    sc1 = axes[row, 0].scatter(x, y, c=original, cmap='jet', s=0.5, alpha=0.8, vmin=0, vmax=1)
    axes[row, 0].set_title(f'Original: {name}', fontweight='bold')
    axes[row, 0].set_aspect('equal')
    axes[row, 0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(sc1, ax=axes[row, 0])
    cbar1.ax.tick_params(labelsize=10)

    sc2 = axes[row, 1].scatter(x, y, c=predicted, cmap='jet', s=0.5, alpha=0.8, vmin=0, vmax=1)
    axes[row, 1].set_title(f'Predicted: {name}', fontweight='bold')
    axes[row, 1].set_aspect('equal')
    axes[row, 1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(sc2, ax=axes[row, 1])
    cbar2.ax.tick_params(labelsize=10)

    sc3 = axes[row, 2].scatter(x, y, c=error, cmap='hot', s=0.5, alpha=0.8)
    axes[row, 2].set_title(f'Absolute Error: {name}', fontweight='bold')
    axes[row, 2].set_aspect('equal')
    axes[row, 2].grid(True, alpha=0.3)
    cbar3 = plt.colorbar(sc3, ax=axes[row, 2])
    cbar3.ax.tick_params(labelsize=10)

plt.suptitle(f'Base Model (64-64-32) Online - PSNR: {psnr:.2f} dB - Timestep: {TIMESTEP}',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/flow_visualization.png", dpi=150, bbox_inches='tight')
print(f"Visualization saved: {OUTPUT_DIR}/flow_visualization.png")
plt.close()

evaluation_results = {
    'model_file': 'base_model_online_final.pth',
    'architecture': '4-64-64-32-4',
    'training_mode': 'online_streaming',
    'parameters': sum(p.numel() for p in model.parameters()),
    'psnr_db': float(psnr),
    'ssim': float(ssim)
}

with open(f"{OUTPUT_DIR}/evaluation_results.json", 'w') as f:
    json.dump(evaluation_results, f, indent=2)

print("Base model online visualization completed")
