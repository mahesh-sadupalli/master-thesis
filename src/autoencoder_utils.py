"""
Autoencoder Utilities for Neural Network-Based Data Compression

This module provides autoencoder models and utilities for compressing
spatio-temporal scientific data. Unlike the coordinate-based INR approach
(which maps coordinates to field values), autoencoders learn compressed
latent representations of the field data itself.

Approach: Temporal-point encoding
- Each spatial point's temporal sequence (all timesteps) is treated as one sample
- Encoder compresses the temporal sequence to a compact latent vector
- Decoder reconstructs the full temporal sequence from the latent vector
- Total storage = model weights + one latent vector per spatial point
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pyarrow.csv as pv
from torcheval.metrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
import time
import os
import json


# Dataset
class TemporalPointDataset(Dataset):
    """
    Dataset that reshapes spatio-temporal data into per-point temporal sequences.

    Each sample contains all field variables across all timesteps for one
    spatial point, flattened into a single vector.

    Input CSV format: x, y, z, t, Vx, Vy, Pressure, TKE (no header)

    Reshaping:
        Raw: (num_points * num_timesteps, 8)
        Per point: (num_timesteps, num_vars) -> flattened to (num_timesteps * num_vars,)
        Dataset: (num_points, num_timesteps * num_vars)

    Args:
        filepath (str): Path to CSV file
    """
    def __init__(self, filepath):
        print(f"Loading dataset from {filepath}")

        read_options = pv.ReadOptions(
            column_names=['x', 'y', 'z', 't', 'Vx', 'Vy', 'Pressure', 'TKE']
        )
        table = pv.read_csv(filepath, read_options=read_options)
        data = table.to_pandas()

        # Sort by spatial location then time for consistent grouping
        data = data.sort_values(['x', 'y', 'z', 't']).reset_index(drop=True)

        coords = data[['x', 'y', 'z']].values.astype(np.float32)
        fields = data[['Vx', 'Vy', 'Pressure', 'TKE']].values.astype(np.float32)

        # Determine grid dimensions
        self.num_timesteps = data['t'].nunique()
        self.num_points = len(data) // self.num_timesteps
        self.num_vars = 4
        self.var_names = ['Vx', 'Vy', 'Pressure', 'TKE']

        print(f"Detected {self.num_points} spatial points x {self.num_timesteps} timesteps")

        # Reshape: (num_points * num_timesteps, num_vars) -> (num_points, num_timesteps, num_vars)
        fields_3d = fields.reshape(self.num_points, self.num_timesteps, self.num_vars)

        # Min-max normalization to [0, 1] (computed globally across all points and timesteps)
        self.field_min = fields.min(axis=0)  # (num_vars,)
        self.field_max = fields.max(axis=0)  # (num_vars,)
        self.field_range = self.field_max - self.field_min
        self.field_range[self.field_range == 0] = 1.0

        fields_norm = (fields_3d - self.field_min) / self.field_range

        # Flatten temporal dimension: (num_points, num_timesteps * num_vars)
        self.input_dim = self.num_timesteps * self.num_vars
        self.data = torch.FloatTensor(fields_norm.reshape(self.num_points, -1))

        # Store unique spatial coordinates for visualization
        self.coords = coords.reshape(self.num_points, self.num_timesteps, 3)[:, 0, :]

        print(f"Dataset ready: {self.num_points} samples, input_dim={self.input_dim}")
        print(f"Field ranges: {dict(zip(self.var_names, [f'{mn:.4f} to {mx:.4f}' for mn, mx in zip(self.field_min, self.field_max)]))}")

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x  # autoencoder: input = target

    def denormalize(self, normalized):
        """
        Convert normalized field values back to original physical scale.

        Args:
            normalized: Tensor of shape (..., num_timesteps * num_vars) or (..., num_vars)

        Returns:
            ndarray: Denormalized values
        """
        if isinstance(normalized, torch.Tensor):
            normalized = normalized.cpu().numpy()

        # If flattened temporal sequence, reshape first
        if normalized.shape[-1] == self.input_dim:
            shape = normalized.shape[:-1] + (self.num_timesteps, self.num_vars)
            normalized = normalized.reshape(shape)
            denorm = normalized * self.field_range + self.field_min
            return denorm.reshape(normalized.shape[:-2] + (-1,))

        # Otherwise assume last dim is num_vars
        return normalized * self.field_range + self.field_min

    def get_normalization_params(self):
        """Return normalization parameters as a serializable dict."""
        return {
            'field_min': self.field_min.tolist(),
            'field_max': self.field_max.tolist(),
            'field_range': self.field_range.tolist(),
            'num_timesteps': self.num_timesteps,
            'num_points': self.num_points,
            'num_vars': self.num_vars,
            'input_dim': self.input_dim
        }


# Models
class LinearEncoder(nn.Module):
    """Fully connected encoder with progressive dimensionality reduction."""
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LinearDecoder(nn.Module):
    """Fully connected decoder that reconstructs from latent space."""
    def __init__(self, latent_dim, hidden_dims, output_dim, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        # No activation on output layer (regression)
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z):
        return self.network(z)


class LinearAutoEncoder(nn.Module):
    """
    Linear (fully connected) autoencoder for spatio-temporal data compression.

    Compresses temporal sequences of field variables at each spatial point
    into compact latent representations.

    Args:
        input_dim (int): Input dimension (num_timesteps * num_vars)
        encoder_hidden (list): Hidden layer sizes for encoder
        decoder_hidden (list): Hidden layer sizes for decoder
        latent_dim (int): Latent space dimension
        dropout (float): Dropout probability
    """
    def __init__(self, input_dim, encoder_hidden, decoder_hidden, latent_dim, dropout=0.1):
        super().__init__()
        self.encoder = LinearEncoder(input_dim, encoder_hidden, latent_dim, dropout)
        self.decoder = LinearDecoder(latent_dim, decoder_hidden, input_dim, dropout)
        self.latent_dim = latent_dim
        self.input_dim = input_dim

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# Model Configurations (Base / Medium / Large)
# input_dim = 300 timesteps * 4 variables = 1200
AE_MODEL_CONFIGS = {
    'base': {
        'encoder_hidden': [256, 128],
        'decoder_hidden': [128, 256],
        'latent_dim': 16,
        'dropout': 0.1,
    },
    'medium': {
        'encoder_hidden': [512, 256, 128],
        'decoder_hidden': [128, 256, 512],
        'latent_dim': 32,
        'dropout': 0.1,
    },
    'large': {
        'encoder_hidden': [512, 256, 128],
        'decoder_hidden': [128, 256, 512],
        'latent_dim': 64,
        'dropout': 0.1,
    },
}


def create_autoencoder(size, input_dim):
    """
    Create a LinearAutoEncoder from a named configuration.

    Args:
        size (str): One of 'base', 'medium', 'large'
        input_dim (int): Input dimension (num_timesteps * num_vars)

    Returns:
        LinearAutoEncoder: Configured model
    """
    config = AE_MODEL_CONFIGS[size]
    return LinearAutoEncoder(
        input_dim=input_dim,
        encoder_hidden=config['encoder_hidden'],
        decoder_hidden=config['decoder_hidden'],
        latent_dim=config['latent_dim'],
        dropout=config['dropout'],
    )


# Metrics (reuses same approach as INR for consistency)
def compute_ae_psnr_ssim(predictions, targets, device):
    """
    Compute PSNR and SSIM for autoencoder reconstructions.

    Reshapes flattened temporal sequences back to (num_points, num_timesteps, num_vars)
    and computes metrics per-timestep, then averages.

    Args:
        predictions (Tensor): Reconstructed data, shape (num_points, num_timesteps * num_vars)
        targets (Tensor): Original data, same shape
        device: PyTorch device

    Returns:
        tuple: (mean_psnr, mean_ssim)
    """
    predictions = predictions.to(device)
    targets = targets.to(device)

    # Compute PSNR on the full flattened data
    psnr_metric = PeakSignalNoiseRatio().to(device)
    psnr_metric.update(predictions, targets)
    psnr = psnr_metric.compute().item()

    # Compute SSIM: reshape to image-like format (1, 1, N, num_vars_per_step)
    # We compute over the full flattened vector treating it as (1, 1, num_points, input_dim)
    pred_ssim = predictions.unsqueeze(0).unsqueeze(0)   # (1, 1, N, D)
    target_ssim = targets.unsqueeze(0).unsqueeze(0)     # (1, 1, N, D)

    ssim_metric = StructuralSimilarityIndexMeasure(
        gaussian_kernel=False,
        kernel_size=1
    ).to(device)

    ssim_metric.update(pred_ssim, target_ssim)
    ssim = ssim_metric.compute().item()

    return psnr, ssim


def compute_ae_relative_error(predictions, targets):
    """
    Compute relative L2 norm error.

    Args:
        predictions (Tensor): Reconstructed data
        targets (Tensor): Original data

    Returns:
        float: Relative error as percentage
    """
    error_norm = torch.norm(predictions - targets)
    target_norm = torch.norm(targets)
    return (error_norm / target_norm * 100).item()


# Compression Ratio Calculation
def compute_compression_ratio(model, dataset, original_size_bytes=None):
    """
    Compute compression ratio for the autoencoder.

    Compressed size = model weights + latent codes for all points.

    Args:
        model: LinearAutoEncoder instance
        dataset: TemporalPointDataset instance
        original_size_bytes (float): Original data size in bytes.
            If None, computed as num_points * num_timesteps * num_vars * 4 bytes.

    Returns:
        dict: Compression statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    model_size_bytes = total_params * 4  # float32

    latent_size_bytes = dataset.num_points * model.latent_dim * 4  # float32

    compressed_size = model_size_bytes + latent_size_bytes

    if original_size_bytes is None:
        original_size_bytes = dataset.num_points * dataset.num_timesteps * dataset.num_vars * 4

    ratio = original_size_bytes / compressed_size

    stats = {
        'total_params': total_params,
        'model_size_kb': model_size_bytes / 1024,
        'latent_size_kb': latent_size_bytes / 1024,
        'compressed_size_kb': compressed_size / 1024,
        'original_size_mb': original_size_bytes / (1024 * 1024),
        'compression_ratio': ratio,
    }
    return stats


# Training
def train_autoencoder(model, train_loader, dataset, device, num_epochs,
                      model_name, output_dir):
    """
    Train autoencoder using offline batch training.

    Args:
        model: LinearAutoEncoder instance
        train_loader: DataLoader for TemporalPointDataset
        dataset: TemporalPointDataset instance
        device: PyTorch device
        num_epochs (int): Number of training epochs
        model_name (str): Identifier for output files
        output_dir (str): Directory for saving results

    Returns:
        dict: Training metrics (loss, psnr, ssim, relative_error, time_per_epoch)
    """
    os.makedirs(output_dir, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    metrics = {
        'loss': [],
        'psnr': [],
        'ssim': [],
        'relative_error': [],
        'time_per_epoch': []
    }

    total_params = sum(p.numel() for p in model.parameters())
    comp_stats = compute_compression_ratio(model, dataset)

    print(f"\nAutoencoder Training Configuration")
    print(f"Model: {model_name}")
    print(f"Spatial points: {dataset.num_points}")
    print(f"Timesteps: {dataset.num_timesteps}")
    print(f"Input dim: {dataset.input_dim} ({dataset.num_timesteps} x {dataset.num_vars})")
    print(f"Latent dim: {model.latent_dim}")
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}")
    print(f"Parameters: {total_params:,}")
    print(f"Model size: {comp_stats['model_size_kb']:.2f} KB")
    print(f"Latent storage: {comp_stats['latent_size_kb']:.2f} KB")
    print(f"Total compressed: {comp_stats['compressed_size_kb']:.2f} KB")
    print(f"Compression ratio: {comp_stats['compression_ratio']:.1f}:1\n")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        model.train()
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)  # outputs, latent
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_predictions.append(outputs.detach())
            all_targets.append(targets)

        epoch_loss /= len(train_loader)
        metrics['loss'].append(epoch_loss)

        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        psnr, ssim = compute_ae_psnr_ssim(all_predictions, all_targets, device)
        rel_error = compute_ae_relative_error(all_predictions, all_targets)

        metrics['psnr'].append(psnr)
        metrics['ssim'].append(ssim)
        metrics['relative_error'].append(rel_error)

        epoch_time = time.time() - epoch_start
        metrics['time_per_epoch'].append(epoch_time)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.6f}, "
                  f"PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, "
                  f"RE={rel_error:.2f}%, Time={epoch_time:.2f}s")

    # Save model
    model_path = os.path.join(output_dir, f'{model_name}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")

    # Save latent codes for all points
    model.eval()
    with torch.no_grad():
        all_data = dataset.data.to(device)
        latent_codes = model.encode(all_data).cpu()
    latent_path = os.path.join(output_dir, f'{model_name}_latent_codes.pt')
    torch.save(latent_codes, latent_path)
    print(f"Latent codes saved: {latent_path} (shape: {latent_codes.shape})")

    # Save normalization parameters
    norm_params = dataset.get_normalization_params()
    norm_path = os.path.join(output_dir, f'{model_name}_normalization.json')
    with open(norm_path, 'w') as f:
        json.dump(norm_params, f, indent=2)
    print(f"Normalization parameters saved: {norm_path}")

    # Print final summary
    print(f"\nTraining completed")
    print(f"Final metrics:")
    print(f"  Loss: {metrics['loss'][-1]:.6f}")
    print(f"  PSNR: {metrics['psnr'][-1]:.2f} dB")
    print(f"  SSIM: {metrics['ssim'][-1]:.4f}")
    print(f"  Relative Error: {metrics['relative_error'][-1]:.2f}%")

    return metrics


def export_metrics_csv(metrics, output_path):
    """Export training metrics to CSV file."""
    df = pd.DataFrame(metrics)
    df['epoch'] = range(1, len(df) + 1)
    df = df[['epoch', 'loss', 'psnr', 'ssim', 'relative_error', 'time_per_epoch']]
    df.to_csv(output_path, index=False)
    print(f"Training metrics saved: {output_path}")
