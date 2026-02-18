"""Evaluation metrics for neural network compression quality."""

import torch
import numpy as np
from torcheval.metrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure


def calculate_metrics_torch(predictions, targets, device):
    """
    Calculate PSNR and SSIM using PyTorch-based metrics.
    For regression data, SSIM is computed on 2D spatial representation.

    Args:
        predictions: Tensor of shape (N, 4) containing normalized predictions
        targets: Tensor of shape (N, 4) containing normalized targets
        device: PyTorch device (CPU or CUDA)

    Returns:
        psnr: Peak Signal-to-Noise Ratio in dB
        ssim: Structural Similarity Index (computed on feature space)
    """
    predictions = predictions.to(device)
    targets = targets.to(device)

    # Calculate PSNR
    psnr_metric = PeakSignalNoiseRatio().to(device)
    psnr_metric.update(predictions, targets)
    psnr = psnr_metric.compute().item()

    # Calculate SSIM on reshaped data
    # Reshape to (batch, channels, height, width) where we treat features as spatial
    # Use sqrt of batch size as spatial dimensions
    n_samples = predictions.shape[0]
    spatial_size = int(np.sqrt(n_samples))

    # Truncate to make it square
    n_use = spatial_size * spatial_size
    pred_square = predictions[:n_use].reshape(1, 4, spatial_size, spatial_size)
    target_square = targets[:n_use].reshape(1, 4, spatial_size, spatial_size)

    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=1.0
    ).to(device)

    ssim = ssim_metric(pred_square, target_square).item()

    return psnr, ssim


def calculate_relative_error(predictions, targets):
    """
    Calculate relative L2 norm error between predictions and targets.

    Args:
        predictions: Tensor of predictions
        targets: Tensor of ground truth values

    Returns:
        Relative error as percentage
    """
    target_norm = torch.norm(targets)
    error = torch.norm(predictions - targets)
    return (error / target_norm * 100).item()
