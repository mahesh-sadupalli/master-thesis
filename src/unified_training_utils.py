"""
Training Utilities for Neural Network-Based Data Compression

This module provides core functionality for training neural networks
to learn compact representations of spatio-temporal data.
Implements min-max normalization and PyTorch-based evaluation metrics.
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


class SpatioTemporalDataset(Dataset):
    """
    PyTorch Dataset for spatio-temporal data with min-max normalization.

    Loads data from CSV and applies min-max scaling to [0, 1] range
    for both input coordinates (x, y, z, t) and target variables (Vx, Vy, P, TKE).

    Args:
        filepath (str): Path to CSV file containing spatio-temporal data
        
    Attributes:
        inputs (ndarray): Normalized input coordinates, shape (N, 4)
        targets (ndarray): Normalized target variables, shape (N, 4)
        input_min, input_max, input_range: Normalization parameters for inputs
        target_min, target_max, target_range: Normalization parameters for targets
    """
    def __init__(self, filepath):
        print(f"Loading dataset from {filepath}")
        
        read_options = pv.ReadOptions(
            column_names=['x', 'y', 'z', 't', 'Vx', 'Vy', 'Pressure', 'TKE']
        )
        table = pv.read_csv(filepath, read_options=read_options)
        data = table.to_pandas().values
        
        self.inputs = data[:, :4].astype(np.float32)
        self.targets = data[:, 4:].astype(np.float32)
        
        print("Applying min-max normalization to [0, 1] range")
        
        self.input_min = self.inputs.min(axis=0)
        self.input_max = self.inputs.max(axis=0)
        self.input_range = self.input_max - self.input_min
        self.input_range[self.input_range == 0] = 1.0
        self.inputs = (self.inputs - self.input_min) / self.input_range
        
        self.target_min = self.targets.min(axis=0)
        self.target_max = self.targets.max(axis=0)
        self.target_range = self.target_max - self.target_min
        self.target_range[self.target_range == 0] = 1.0
        self.targets = (self.targets - self.target_min) / self.target_range
        
        print(f"Dataset loaded: {len(self)} samples")
        print(f"Input shape: {self.inputs.shape}, Target shape: {self.targets.shape}")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.targets[idx])
    
    def denormalize_inputs(self, normalized):
        """
        Convert normalized input coordinates back to original physical space.
        
        Args:
            normalized: Normalized coordinates as tensor or ndarray
            
        Returns:
            ndarray: Denormalized coordinates in original units
        """
        if isinstance(normalized, torch.Tensor):
            normalized = normalized.cpu().numpy()
        return normalized * self.input_range + self.input_min
    
    def denormalize_targets(self, normalized):
        """
        Convert normalized target variables back to original physical space.
        
        Args:
            normalized: Normalized variables as tensor or ndarray
            
        Returns:
            ndarray: Denormalized variables in original units
        """
        if isinstance(normalized, torch.Tensor):
            normalized = normalized.cpu().numpy()
        return normalized * self.target_range + self.target_min


class BaseCompressor(nn.Module):
    """
    Base compressor for spatio-temporal regression.
    Architecture: 4 inputs → 64 → 64 → 32 → 4 outputs

    Maps spatio-temporal coordinates (x, y, z, t) to target variables (Vx, Vy, P, TKE).
    Total parameters: 6,692
    """
    def __init__(self):
        super(BaseCompressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
    
    def forward(self, x):
        return self.network(x)


class LargeCompressor(nn.Module):
    """
    Large compressor for spatio-temporal regression.
    Architecture: 4 inputs → 128 → 128 → 64 → 4 outputs

    Maps spatio-temporal coordinates (x, y, z, t) to target variables (Vx, Vy, P, TKE).
    Larger capacity for improved reconstruction accuracy.
    Total parameters: 25,668
    """
    def __init__(self):
        super(LargeCompressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
    
    def forward(self, x):
        return self.network(x)


class MediumCompressor(nn.Module):
    """
    Medium compressor for spatio-temporal regression.
    Architecture: 4 inputs → 96 → 96 → 48 → 4 outputs

    Maps spatio-temporal coordinates (x, y, z, t) to target variables (Vx, Vy, P, TKE).
    Balanced capacity between base and large models.
    Total parameters: 14,644
    """
    def __init__(self):
        super(MediumCompressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 4)
        )
    
    def forward(self, x):
        return self.network(x)


def compute_psnr_ssim(predictions, targets, device):
    """
    Compute Peak Signal-to-Noise Ratio and Structural Similarity Index.
    
    Uses torchmetrics with gaussian_kernel=False to avoid padding issues
    on regression data with small feature dimensions.
    
    Args:
        predictions (Tensor): Predicted values, shape (N, 4)
        targets (Tensor): Ground truth values, shape (N, 4)
        device: PyTorch device (CPU or CUDA)
    
    Returns:
        tuple: (psnr, ssim) where:
            psnr (float): Peak Signal-to-Noise Ratio in dB
            ssim (float): Structural Similarity Index in [0, 1]
    """
    predictions = predictions.to(device)
    targets = targets.to(device)
    
    # Compute PSNR
    psnr_metric = PeakSignalNoiseRatio().to(device)
    psnr_metric.update(predictions, targets)
    psnr = psnr_metric.compute().item()
    
    # Compute SSIM with proper reshaping for regression data
    # Reshape: (N, 4) -> (N, 1, 1, 4)
    pred_ssim = predictions.view(-1, 1, 1, predictions.shape[1])
    target_ssim = targets.view(-1, 1, 1, targets.shape[1])
    
    # Permute: (N, 1, 1, 4) -> (1, 1, N, 4)
    pred_ssim = pred_ssim.permute(1, 2, 0, 3)
    target_ssim = target_ssim.permute(1, 2, 0, 3)
    
    # Disable Gaussian kernel to avoid padding issues with small width dimension
    ssim_metric = StructuralSimilarityIndexMeasure(
        gaussian_kernel=False,
        kernel_size=1
    ).to(device)
    
    ssim_metric.update(pred_ssim, target_ssim)
    ssim = ssim_metric.compute().item()
    
    return psnr, ssim


def compute_relative_error(predictions, targets):
    """
    Compute relative L2 norm error between predictions and targets.
    
    Calculated as ||predictions - targets||_2 / ||targets||_2 * 100
    
    Args:
        predictions (Tensor): Predicted values
        targets (Tensor): Ground truth values
    
    Returns:
        float: Relative error as percentage
    """
    pred_norm = torch.norm(predictions)
    target_norm = torch.norm(targets)
    error_norm = torch.norm(predictions - targets)
    return (error_norm / target_norm * 100).item()


def train_model(model, train_loader, dataset, device, num_epochs,
                model_name, output_dir):
    """
    Train neural network using offline batch training.
    
    Performs full dataset training for specified number of epochs with
    continuous metric monitoring and checkpoint saving.
    
    Args:
        model: Neural network model
        train_loader: PyTorch DataLoader for training data
        dataset: SpatioTemporalDataset instance for normalization parameters
        device: PyTorch device
        num_epochs (int): Number of training epochs
        model_name (str): Identifier for output files
        output_dir (str): Directory path for saving results
    
    Returns:
        dict: Training metrics containing loss, PSNR, SSIM, relative error,
              and time per epoch for all epochs
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
    print(f"\nOffline training configuration")
    print(f"Model: {model_name}")
    print(f"Training samples: {len(dataset)}")
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}")
    print(f"Parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024:.2f} KB\n")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        model.train()
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            all_predictions.append(outputs.detach())
            all_targets.append(targets)
        
        epoch_loss /= len(train_loader)
        metrics['loss'].append(epoch_loss)
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        psnr, ssim = compute_psnr_ssim(all_predictions, all_targets, device)
        rel_error = compute_relative_error(all_predictions, all_targets)
        
        metrics['psnr'].append(psnr)
        metrics['ssim'].append(ssim)
        metrics['relative_error'].append(rel_error)
        
        epoch_time = time.time() - epoch_start
        metrics['time_per_epoch'].append(epoch_time)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.6f}, "
                  f"PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, "
                  f"RE={rel_error:.2f}%, Time={epoch_time:.2f}s")
    
    model_path = os.path.join(output_dir, f'{model_name}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")
    
    norm_params = {
        'input_min': dataset.input_min.tolist(),
        'input_max': dataset.input_max.tolist(),
        'input_range': dataset.input_range.tolist(),
        'target_min': dataset.target_min.tolist(),
        'target_max': dataset.target_max.tolist(),
        'target_range': dataset.target_range.tolist()
    }
    
    norm_path = os.path.join(output_dir, f'{model_name}_normalization.json')
    with open(norm_path, 'w') as f:
        json.dump(norm_params, f, indent=2)
    print(f"Normalization parameters saved: {norm_path}")
    
    print(f"\nTraining completed")
    print(f"Final metrics:")
    print(f"  Loss: {metrics['loss'][-1]:.6f}")
    print(f"  PSNR: {metrics['psnr'][-1]:.2f} dB")
    print(f"  SSIM: {metrics['ssim'][-1]:.4f}")
    print(f"  Relative Error: {metrics['relative_error'][-1]:.2f}%")
    print(f"PSNR improvement: {metrics['psnr'][0]:.2f} → {metrics['psnr'][-1]:.2f} dB "
          f"(+{metrics['psnr'][-1]-metrics['psnr'][0]:.2f} dB)\n")
    
    return metrics


def export_metrics_csv(metrics, output_path):
    """
    Export training metrics to CSV file.
    
    Args:
        metrics (dict): Dictionary containing metric arrays
        output_path (str): Output CSV file path
    """
    df = pd.DataFrame(metrics)
    df['epoch'] = range(1, len(df) + 1)
    df = df[['epoch', 'loss', 'psnr', 'ssim', 'relative_error', 'time_per_epoch']]
    df.to_csv(output_path, index=False)
    print(f"Training metrics saved: {output_path}")
