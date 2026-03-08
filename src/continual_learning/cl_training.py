"""
Continual Learning Training Loop for Online Neural Compression

Modified version of train_online that accepts a continual learning strategy
object to mitigate catastrophic forgetting. Imports all models, dataset,
and metrics from the existing unified_training_utils.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import time

# Add parent directory to path for imports from existing code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from unified_training_utils import compute_psnr_ssim, compute_relative_error


def train_online_cl(model, dataset, device, epochs_per_window, model_name,
                    output_dir, strategy, num_windows=20):
    """
    Train neural network using online streaming with a continual learning strategy.

    This is the core training loop that accepts a strategy object to modify
    the loss computation and add forgetting mitigation mechanisms.

    Args:
        model: Neural network model (BaseCompressor, MediumCompressor, or LargeCompressor)
        dataset: SpatioTemporalDataset instance
        device: PyTorch device
        epochs_per_window (int): Number of epochs to train on each window
        model_name (str): Identifier for output files
        output_dir (str): Directory path for saving results
        strategy: Continual learning strategy object (from cl_strategies)
        num_windows (int): Number of temporal windows

    Returns:
        dict: Training metrics for all windows
    """
    os.makedirs(output_dir, exist_ok=True)

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
    print(f"\n[Training Configuration]")
    print(f"  Strategy     : {strategy.name}")
    print(f"  Model        : {model_name} ({total_params:,} parameters, {total_params * 4 / 1024:.1f} KB)")
    print(f"  Windows      : {num_windows}")
    print(f"  Epochs/window: {epochs_per_window}")
    print(f"  Device       : {device}")
    print(f"  Hyperparams  : {strategy.get_config()}")

    # Get unique timesteps and divide into windows
    coords_denorm = dataset.denormalize_inputs(dataset.inputs)
    unique_times = np.unique(coords_denorm[:, 3])
    times_per_window = len(unique_times) // num_windows

    print(f"  Timesteps    : {len(unique_times)} total, {times_per_window} per window\n")

    total_start = time.time()

    for window_idx in range(num_windows):
        window_start = time.time()

        # Select timesteps for this window
        start_idx = window_idx * times_per_window
        end_idx = start_idx + times_per_window if window_idx < num_windows - 1 else len(unique_times)
        window_times = unique_times[start_idx:end_idx]

        # Get data for this window
        time_mask = np.isin(coords_denorm[:, 3], window_times)
        window_inputs = torch.from_numpy(dataset.inputs[time_mask]).to(device)
        window_targets = torch.from_numpy(dataset.targets[time_mask]).to(device)

        # Strategy pre-window hook
        strategy.before_window(model, window_idx, window_inputs, window_targets, device)

        # Train on this window
        model.train()
        for epoch in range(epochs_per_window):
            optimizer.zero_grad()

            outputs = model(window_inputs)
            loss = strategy.compute_loss(
                model, criterion, outputs, targets=window_targets,
                window_inputs=window_inputs, device=device
            )
            loss.backward()
            optimizer.step()

        # Strategy post-window hook
        strategy.after_window(model, window_idx, window_inputs, window_targets, device)

        # Evaluate on this window
        model.eval()
        with torch.no_grad():
            predictions = model(window_inputs)

        loss_val = criterion(predictions, window_targets).item()
        psnr, ssim = compute_psnr_ssim(predictions, window_targets, device)
        rel_error = compute_relative_error(predictions, window_targets)

        window_time = time.time() - window_start

        metrics['window'].append(window_idx + 1)
        metrics['loss'].append(loss_val)
        metrics['psnr'].append(psnr)
        metrics['ssim'].append(ssim)
        metrics['relative_error'].append(rel_error)
        metrics['time_per_window'].append(window_time)

        print(f"  Window {window_idx+1:3d}/{num_windows}: "
              f"PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, "
              f"RE={rel_error:.2f}%, Time={window_time:.2f}s")

    total_time = time.time() - total_start

    # Save final model
    final_model_path = os.path.join(output_dir, f'{model_name}_final.pth')
    torch.save(model.state_dict(), final_model_path)

    # Save normalization parameters
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

    # Save training summary
    summary = {
        'model_name': model_name,
        'strategy': strategy.get_config(),
        'total_windows': num_windows,
        'epochs_per_window': epochs_per_window,
        'total_epochs': num_windows * epochs_per_window,
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

    print(f"\n[Training Complete] {strategy.name}")
    print(f"  Final PSNR   : {metrics['psnr'][-1]:.2f} dB")
    print(f"  Final SSIM   : {metrics['ssim'][-1]:.4f}")
    print(f"  Final RE     : {metrics['relative_error'][-1]:.2f}%")
    print(f"  Total time   : {total_time:.2f}s")
    print(f"  Model saved  : {final_model_path}\n")

    return metrics


def evaluate_full_dataset(model, dataset, device, model_name=None):
    """
    Evaluate a trained model on the entire dataset.

    This is the critical evaluation that reveals catastrophic forgetting —
    it tests reconstruction quality across ALL timesteps, not just the last window.

    Args:
        model: Trained neural network model
        dataset: SpatioTemporalDataset instance
        device: PyTorch device
        model_name (str, optional): For display purposes

    Returns:
        dict: Full-dataset evaluation metrics
    """
    model.eval()

    all_inputs = torch.from_numpy(dataset.inputs).to(device)
    all_targets = torch.from_numpy(dataset.targets).to(device)

    with torch.no_grad():
        all_predictions = model(all_inputs)

    criterion = nn.MSELoss()
    loss = criterion(all_predictions, all_targets).item()
    psnr, ssim = compute_psnr_ssim(all_predictions, all_targets, device)
    rel_error = compute_relative_error(all_predictions, all_targets)

    results = {
        'loss': loss,
        'psnr_db': psnr,
        'ssim': ssim,
        'relative_error_pct': rel_error
    }

    label = model_name if model_name else "model"
    print(f"[Full Dataset Evaluation] {label}")
    print(f"  PSNR : {psnr:.2f} dB | SSIM : {ssim:.4f} | RE : {rel_error:.2f}%")

    return results
