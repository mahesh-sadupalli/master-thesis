"""
Continual Learning Training Loop for Online Autoencoder Compression

Adapts the CL training pipeline for autoencoders. Key differences from
the INR training loop (cl_training.py):
  - Uses WindowedAEDataset with get_window_data() instead of time masking
  - Wraps AE model so forward() returns only x_hat (strategies expect single tensor)
  - Uses AE-specific metrics (compute_ae_psnr_ssim, compute_ae_relative_error)
  - Full-dataset evaluation reconstructs each window separately then concatenates
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from autoencoder_utils import compute_ae_psnr_ssim, compute_ae_relative_error


class AEForwardWrapper(nn.Module):
    """
    Wraps LinearAutoEncoder so forward() returns only the reconstruction.

    CL strategies call model(inputs) internally (e.g., for replay samples)
    and expect a single tensor, not a (x_hat, z) tuple. This wrapper ensures
    compatibility with all existing strategies without modifying them.
    """

    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, x):
        x_hat, _ = self.ae(x)
        return x_hat


def train_online_ae_cl(model, dataset, device, epochs_per_window, model_name,
                       output_dir, strategy, num_windows=20):
    """
    Train autoencoder using online streaming with a continual learning strategy.

    Args:
        model: LinearAutoEncoder instance (will be wrapped for strategy compatibility)
        dataset: WindowedAEDataset instance
        device: PyTorch device
        epochs_per_window (int): Number of epochs per window
        model_name (str): Identifier for output files
        output_dir (str): Directory for saving results
        strategy: CL strategy object (NaiveStrategy, ExperienceReplayStrategy, etc.)
        num_windows (int): Number of temporal windows

    Returns:
        dict: Training metrics for all windows
    """
    os.makedirs(output_dir, exist_ok=True)

    # Wrap model so strategies get single-tensor outputs
    wrapped_model = AEForwardWrapper(model).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    metrics = {
        'window': [],
        'loss': [],
        'psnr': [],
        'ssim': [],
        'relative_error': [],
        'time_per_window': []
    }

    total_params = sum(p.numel() for p in model.parameters())
    print("\n[AE Online Training Configuration]")
    print("  Strategy     : {}".format(strategy.name))
    print("  Model        : {} ({:,} parameters, {:.1f} KB)".format(
        model_name, total_params, total_params * 4 / 1024))
    print("  Input dim    : {} ({} timesteps x {} vars)".format(
        dataset.window_input_dim, dataset.time_seq, dataset.num_vars))
    print("  Latent dim   : {}".format(model.latent_dim))
    print("  Windows      : {}".format(num_windows))
    print("  Epochs/window: {}".format(epochs_per_window))
    print("  Device       : {}".format(device))
    print("  Hyperparams  : {}".format(strategy.get_config()))

    print("  Spatial pts  : {:,}".format(dataset.num_points))
    print("  Timesteps/win: {}\n".format(dataset.time_seq))

    total_start = time.time()

    for window_idx in range(num_windows):
        window_start = time.time()

        # Get data for this window
        window_data = dataset.get_window_data(window_idx).to(device)
        # AE: input = target
        window_inputs = window_data
        window_targets = window_data

        # Strategy pre-window hook (uses wrapped model)
        strategy.before_window(wrapped_model, window_idx,
                               window_inputs, window_targets, device)

        # Train on this window
        wrapped_model.train()
        for epoch in range(epochs_per_window):
            optimizer.zero_grad()

            outputs = wrapped_model(window_inputs)
            loss = strategy.compute_loss(
                wrapped_model, criterion, outputs, targets=window_targets,
                window_inputs=window_inputs, device=device
            )
            loss.backward()
            optimizer.step()

        # Strategy post-window hook
        strategy.after_window(wrapped_model, window_idx,
                              window_inputs, window_targets, device)

        # Evaluate on this window
        wrapped_model.eval()
        with torch.no_grad():
            predictions = wrapped_model(window_inputs)

        loss_val = criterion(predictions, window_targets).item()
        psnr, ssim = compute_ae_psnr_ssim(predictions, window_targets, device)
        rel_error = compute_ae_relative_error(predictions, window_targets)

        window_time = time.time() - window_start

        metrics['window'].append(window_idx + 1)
        metrics['loss'].append(loss_val)
        metrics['psnr'].append(psnr)
        metrics['ssim'].append(ssim)
        metrics['relative_error'].append(rel_error)
        metrics['time_per_window'].append(window_time)

        print("  Window {:3d}/{}: "
              "PSNR={:.2f} dB, SSIM={:.4f}, "
              "RE={:.2f}%, Time={:.2f}s".format(
                  window_idx + 1, num_windows,
                  psnr, ssim, rel_error, window_time))

    total_time = time.time() - total_start

    # Save final model (unwrapped AE state dict)
    final_model_path = os.path.join(output_dir, '{}_final.pth'.format(model_name))
    torch.save(model.state_dict(), final_model_path)

    # Save normalization parameters
    norm_params = dataset.get_normalization_params()
    norm_path = os.path.join(output_dir, '{}_normalization.json'.format(model_name))
    with open(norm_path, 'w') as f:
        json.dump(norm_params, f, indent=2)

    # Save training summary
    summary = {
        'model_name': model_name,
        'strategy': strategy.get_config(),
        'total_windows': num_windows,
        'epochs_per_window': epochs_per_window,
        'total_epochs': num_windows * epochs_per_window,
        'input_dim': dataset.window_input_dim,
        'latent_dim': model.latent_dim,
        'parameters': total_params,
        'final_loss': metrics['loss'][-1],
        'final_psnr': metrics['psnr'][-1],
        'final_ssim': metrics['ssim'][-1],
        'final_relative_error': metrics['relative_error'][-1],
        'total_training_time': total_time,
        'avg_time_per_window': total_time / num_windows
    }

    summary_path = os.path.join(output_dir, 'online_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n[Training Complete] {}".format(strategy.name))
    print("  Final PSNR   : {:.2f} dB".format(metrics['psnr'][-1]))
    print("  Final SSIM   : {:.4f}".format(metrics['ssim'][-1]))
    print("  Final RE     : {:.2f}%".format(metrics['relative_error'][-1]))
    print("  Total time   : {:.2f}s".format(total_time))
    print("  Model saved  : {}\n".format(final_model_path))

    return metrics


def evaluate_full_dataset_ae(model, dataset, device, model_name=None):
    """
    Evaluate a trained online AE on the entire dataset.

    Since the model takes (time_seq * num_vars)-dim input, it evaluates
    each window separately, then computes global metrics by concatenating
    all predictions and targets.

    This reveals catastrophic forgetting: the model was last trained on
    window 20, so earlier windows will have degraded reconstruction.

    Args:
        model: Trained LinearAutoEncoder
        dataset: WindowedAEDataset instance
        device: PyTorch device
        model_name (str, optional): For display

    Returns:
        dict: Full-dataset evaluation metrics
    """
    wrapped = AEForwardWrapper(model).to(device)
    wrapped.eval()

    all_predictions = []
    all_targets = []

    for window_idx in range(dataset.num_windows):
        window_data = dataset.get_window_data(window_idx).to(device)

        with torch.no_grad():
            preds = wrapped(window_data)

        all_predictions.append(preds)
        all_targets.append(window_data)

    # Concatenate across windows: (num_points * num_windows, window_input_dim)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    criterion = nn.MSELoss()
    loss = criterion(all_predictions, all_targets).item()
    psnr, ssim = compute_ae_psnr_ssim(all_predictions, all_targets, device)
    rel_error = compute_ae_relative_error(all_predictions, all_targets)

    results = {
        'loss': loss,
        'psnr_db': psnr,
        'ssim': ssim,
        'relative_error_pct': rel_error
    }

    label = model_name if model_name else "AE model"
    print("[Full Dataset Evaluation] {}".format(label))
    print("  PSNR : {:.2f} dB | SSIM : {:.4f} | RE : {:.2f}%".format(
        psnr, ssim, rel_error))

    return results
