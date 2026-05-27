"""
Train Base Linear AutoEncoder for spatio-temporal data compression.

Architecture: 1200 → 256 → 128 → [16] → 128 → 256 → 1200
Latent dim: 16
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from autoencoder_utils import (
    TemporalPointDataset,
    create_autoencoder,
    train_autoencoder,
    export_metrics_csv,
    compute_compression_ratio,
)
from torch.utils.data import DataLoader

# ---- Configuration ----
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ML_test_loader_original_data.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'base_autoencoder')
MODEL_NAME = 'base_linear_ae'
MODEL_SIZE = 'base'
NUM_EPOCHS = 150
BATCH_SIZE = 512

# ---- Setup ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Load Data ----
dataset = TemporalPointDataset(DATA_PATH)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- Create Model ----
model = create_autoencoder(MODEL_SIZE, dataset.input_dim).to(device)

# ---- Train ----
metrics = train_autoencoder(
    model=model,
    train_loader=train_loader,
    dataset=dataset,
    device=device,
    num_epochs=NUM_EPOCHS,
    model_name=MODEL_NAME,
    output_dir=OUTPUT_DIR,
)

# ---- Export Metrics ----
export_metrics_csv(metrics, os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_metrics.csv'))

# ---- Print Compression Stats ----
comp = compute_compression_ratio(model, dataset)
print(f"\nCompression Summary:")
print(f"  Model parameters: {comp['total_params']:,}")
print(f"  Model size: {comp['model_size_kb']:.2f} KB")
print(f"  Latent codes: {comp['latent_size_kb']:.2f} KB")
print(f"  Total compressed: {comp['compressed_size_kb']:.2f} KB ({comp['compressed_size_kb']/1024:.2f} MB)")
print(f"  Original data: {comp['original_size_mb']:.2f} MB")
print(f"  Compression ratio: {comp['compression_ratio']:.1f}:1")
