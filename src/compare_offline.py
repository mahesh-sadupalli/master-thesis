"""
Offline Training — Model Comparison Plots

Loads saved metrics CSVs from all 3 offline models (Base, Medium, Large)
and generates 4 comparison figures with all models overlaid on the same axes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "../results"
OUTPUT_DIR = "../results/comparison_offline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load metrics from all 3 models
base = pd.read_csv(f"{RESULTS_DIR}/base_model_offline/base_model_metrics.csv")
medium = pd.read_csv(f"{RESULTS_DIR}/medium_model_offline/medium_model_metrics.csv")
large = pd.read_csv(f"{RESULTS_DIR}/large_model_offline/large_model_metrics.csv")

models = [
    ('Base (64-64-32)', base, 'tab:blue'),
    ('Medium (96-96-48)', medium, 'tab:orange'),
    ('Large (128-128-64)', large, 'tab:green'),
]

# 1. MSE Loss Comparison
fig, ax = plt.subplots(figsize=(12, 6))
for label, df, color in models:
    ax.plot(df['epoch'], df['loss'], linewidth=2, color=color, label=label)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('MSE Loss Comparison — Offline Training', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/loss_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/loss_comparison.png")
plt.close()

# 2. PSNR Comparison
fig, ax = plt.subplots(figsize=(12, 6))
for label, df, color in models:
    ax.plot(df['epoch'], df['psnr'], linewidth=2, color=color, label=label)
ax.set_xlabel('Epoch')
ax.set_ylabel('PSNR (dB)')
ax.set_title('Peak Signal to Noise Ratio (PSNR) Comparison — Offline Training', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/psnr_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/psnr_comparison.png")
plt.close()

# 3. SSIM Comparison
fig, ax = plt.subplots(figsize=(12, 6))
for label, df, color in models:
    ax.plot(df['epoch'], df['ssim'], linewidth=2, color=color, label=label)
ax.set_xlabel('Epoch')
ax.set_ylabel('SSIM')
ax.set_title('Structural Similarity Index (SSIM) Comparison — Offline Training', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ssim_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/ssim_comparison.png")
plt.close()

# 4. Relative Error Comparison
fig, ax = plt.subplots(figsize=(12, 6))
for label, df, color in models:
    ax.plot(df['epoch'], df['relative_error'], linewidth=2, color=color, label=label)
ax.set_xlabel('Epoch')
ax.set_ylabel('Relative Error (%)')
ax.set_title('Relative Error Comparison — Offline Training', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/relative_error_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/relative_error_comparison.png")
plt.close()

print(f"\nAll comparison plots saved to {OUTPUT_DIR}/")
