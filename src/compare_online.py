"""
Online Training — Model Comparison Plots

Loads saved metrics CSVs from all 3 online models (Base, Medium, Large)
and generates 4 comparison figures with all models overlaid on the same axes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "../results"
OUTPUT_DIR = "../results/comparison_online"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load metrics from all 3 models
base = pd.read_csv(f"{RESULTS_DIR}/base_model_online/training_metrics.csv")
medium = pd.read_csv(f"{RESULTS_DIR}/medium_model_online/training_metrics.csv")
large = pd.read_csv(f"{RESULTS_DIR}/large_model_online/training_metrics.csv")

models = [
    ('Base (64-64-32)', base, 'tab:blue'),
    ('Medium (96-96-48)', medium, 'tab:orange'),
    ('Large (128-128-64)', large, 'tab:green'),
]

# 1. MSE Loss Comparison
fig, ax = plt.subplots(figsize=(12, 6))
for label, df, color in models:
    ax.plot(df['window'], df['loss'], '-o', linewidth=2, markersize=4, color=color, label=label)
ax.set_xlabel('Window')
ax.set_ylabel('MSE Loss')
ax.set_title('MSE Loss Comparison — Online Training', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/loss_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/loss_comparison.png")
plt.close()

# 2. PSNR Comparison
fig, ax = plt.subplots(figsize=(12, 6))
for label, df, color in models:
    ax.plot(df['window'], df['psnr'], '-o', linewidth=2, markersize=4, color=color, label=label)
ax.set_xlabel('Window')
ax.set_ylabel('PSNR (dB)')
ax.set_title('Peak Signal to Noise Ratio (PSNR) Comparison — Online Training', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/psnr_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/psnr_comparison.png")
plt.close()

# 3. SSIM Comparison
fig, ax = plt.subplots(figsize=(12, 6))
for label, df, color in models:
    ax.plot(df['window'], df['ssim'], '-o', linewidth=2, markersize=4, color=color, label=label)
ax.set_xlabel('Window')
ax.set_ylabel('SSIM')
ax.set_title('Structural Similarity Index (SSIM) Comparison — Online Training', fontweight='bold')
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
    ax.plot(df['window'], df['relative_error'], '-o', linewidth=2, markersize=4, color=color, label=label)
ax.set_xlabel('Window')
ax.set_ylabel('Relative Error (%)')
ax.set_title('Relative Error Comparison — Online Training', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/relative_error_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/relative_error_comparison.png")
plt.close()

print(f"\nAll comparison plots saved to {OUTPUT_DIR}/")
