"""Online (streaming) training loop with temporal windows."""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import time

from .metrics import calculate_metrics_torch, calculate_relative_error


def train_online(model, dataset, device, epochs_per_window, model_name, output_dir, num_windows=20):
    """
    Train neural network using online streaming approach.
    Divides data into temporal windows and trains incrementally.

    Args:
        model: Neural network model
        dataset: FlowDatasetMinMax instance
        device: PyTorch device
        epochs_per_window: Number of epochs to train on each window
        model_name: Name identifier for saving outputs
        output_dir: Directory path for saving results
        num_windows: Number of temporal windows to divide data into

    Returns:
        all_metrics: Dictionary containing training history across all windows
    """
    os.makedirs(output_dir, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    all_metrics = {
        'window': [],
        'loss': [],
        'psnr': [],
        'ssim': [],
        'relative_error': [],
        'time_per_window': []
    }

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nOnline training: {model_name}")
    print(f"Total samples: {len(dataset)}")
    print(f"Windows: {num_windows}")
    print(f"Epochs per window: {epochs_per_window}")
    print(f"Device: {device}")
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024:.2f} KB\n")

    # Get unique timesteps
    coords_denorm = dataset.denormalize_input(dataset.inputs)
    unique_times = np.unique(coords_denorm[:, 3])
    print(f"Total unique timesteps: {len(unique_times)}")

    # Divide timesteps into windows
    times_per_window = len(unique_times) // num_windows
    print(f"Timesteps per window: {times_per_window}\n")

    for window_idx in range(num_windows):
        window_start_time = time.time()

        # Select timesteps for this window
        start_idx = window_idx * times_per_window
        end_idx = start_idx + times_per_window if window_idx < num_windows - 1 else len(unique_times)
        window_times = unique_times[start_idx:end_idx]

        # Get data for this window
        time_mask = np.isin(coords_denorm[:, 3], window_times)
        window_inputs = torch.from_numpy(dataset.inputs[time_mask]).to(device)
        window_targets = torch.from_numpy(dataset.targets[time_mask]).to(device)

        print(f"Window {window_idx+1}/{num_windows}: {len(window_inputs)} samples, "
              f"timesteps {window_times[0]:.4f}-{window_times[-1]:.4f}")

        # Train on this window
        for epoch in range(epochs_per_window):
            model.train()
            optimizer.zero_grad()

            outputs = model(window_inputs)
            loss = criterion(outputs, window_targets)
            loss.backward()
            optimizer.step()

        # Evaluate on this window
        model.eval()
        with torch.no_grad():
            predictions = model(window_inputs)

        loss_val = criterion(predictions, window_targets).item()
        psnr, ssim = calculate_metrics_torch(predictions, window_targets, device)
        rel_error = calculate_relative_error(predictions, window_targets)

        window_time = time.time() - window_start_time

        all_metrics['window'].append(window_idx + 1)
        all_metrics['loss'].append(loss_val)
        all_metrics['psnr'].append(psnr)
        all_metrics['ssim'].append(ssim)
        all_metrics['relative_error'].append(rel_error)
        all_metrics['time_per_window'].append(window_time)

        print(f"  Loss={loss_val:.6f}, PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, "
              f"RE={rel_error:.2f}%, Time={window_time:.2f}s")

        # Save window checkpoint
        window_dir = os.path.join(output_dir, f'window_{window_idx+1:02d}')
        os.makedirs(window_dir, exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(window_dir, f'{model_name}_window_{window_idx+1:02d}.pth'))

    # Save final model
    final_model_path = os.path.join(output_dir, f'{model_name}_online_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved: {final_model_path}")

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
    print(f"Normalization parameters saved: {norm_path}")

    # Save summary
    summary = {
        'model_name': model_name,
        'total_windows': num_windows,
        'epochs_per_window': epochs_per_window,
        'total_epochs': num_windows * epochs_per_window,
        'final_loss': all_metrics['loss'][-1],
        'final_psnr': all_metrics['psnr'][-1],
        'final_ssim': all_metrics['ssim'][-1],
        'final_relative_error': all_metrics['relative_error'][-1],
        'total_training_time': sum(all_metrics['time_per_window']),
        'avg_time_per_window': np.mean(all_metrics['time_per_window'])
    }

    summary_path = os.path.join(output_dir, 'online_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nOnline training completed")
    print(f"Total windows: {num_windows}")
    print(f"Total epochs: {num_windows * epochs_per_window}")
    print(f"Final Loss: {all_metrics['loss'][-1]:.6f}")
    print(f"Final PSNR: {all_metrics['psnr'][-1]:.2f} dB")
    print(f"Final SSIM: {all_metrics['ssim'][-1]:.4f}")
    print(f"Total time: {sum(all_metrics['time_per_window']):.2f}s\n")

    return all_metrics
