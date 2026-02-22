"""
Online Training Utilities for Streaming Data Compression

Implements temporal window-based training to simulate in-situ compression
where the model learns incrementally from streaming spatio-temporal data.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import time
from unified_training_utils import compute_psnr_ssim, compute_relative_error


def train_online(model, dataset, device, epochs_per_window, model_name,
                 output_dir, num_windows=20):
    """
    Train neural network using online streaming approach with temporal windows.
    
    Divides dataset into temporal windows and trains incrementally to simulate
    in-situ compression where data arrives as a stream during simulation.
    
    Args:
        model: Neural network model
        dataset: SpatioTemporalDataset instance
        device: PyTorch device
        epochs_per_window (int): Number of epochs to train on each window
        model_name (str): Identifier for output files
        output_dir (str): Directory path for saving results
        num_windows (int): Number of temporal windows to divide data into
    
    Returns:
        dict: Training metrics containing loss, PSNR, SSIM, relative error,
              and time per window for all windows
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
    print(f"\nOnline training configuration")
    print(f"Model: {model_name}")
    print(f"Total samples: {len(dataset)}")
    print(f"Windows: {num_windows}")
    print(f"Epochs per window: {epochs_per_window}")
    print(f"Total epochs: {num_windows * epochs_per_window}")
    print(f"Device: {device}")
    print(f"Parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024:.2f} KB\n")
    
    # Get unique timesteps and divide into windows
    coords_denorm = dataset.denormalize_inputs(dataset.inputs)
    unique_times = np.unique(coords_denorm[:, 3])
    times_per_window = len(unique_times) // num_windows
    
    print(f"Total unique timesteps: {len(unique_times)}")
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
        
        print(f"Window {window_idx+1}/{num_windows}: {len(window_inputs):,} samples, "
              f"timesteps [{window_times[0]:.4f}, {window_times[-1]:.4f}]")
        
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
        psnr, ssim = compute_psnr_ssim(predictions, window_targets, device)
        rel_error = compute_relative_error(predictions, window_targets)
        
        window_time = time.time() - window_start_time
        
        metrics['window'].append(window_idx + 1)
        metrics['loss'].append(loss_val)
        metrics['psnr'].append(psnr)
        metrics['ssim'].append(ssim)
        metrics['relative_error'].append(rel_error)
        metrics['time_per_window'].append(window_time)
        
        print(f"  Loss={loss_val:.6f}, PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, "
              f"RE={rel_error:.2f}%, Time={window_time:.2f}s")
        
        # Save window checkpoint
        window_dir = os.path.join(output_dir, f'window_{window_idx+1:02d}')
        os.makedirs(window_dir, exist_ok=True)
        torch.save(model.state_dict(), 
                   os.path.join(window_dir, f'{model_name}_window_{window_idx+1:02d}.pth'))
    
    # Save final model
    final_model_path = os.path.join(output_dir, f'{model_name}_final.pth')
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
    
    # Save training summary
    summary = {
        'model_name': model_name,
        'total_windows': num_windows,
        'epochs_per_window': epochs_per_window,
        'total_epochs': num_windows * epochs_per_window,
        'final_loss': metrics['loss'][-1],
        'final_psnr': metrics['psnr'][-1],
        'final_ssim': metrics['ssim'][-1],
        'final_relative_error': metrics['relative_error'][-1],
        'total_training_time': sum(metrics['time_per_window']),
        'avg_time_per_window': np.mean(metrics['time_per_window'])
    }
    
    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nOnline training completed")
    print(f"Total windows: {num_windows}")
    print(f"Total epochs: {num_windows * epochs_per_window}")
    print(f"Final metrics:")
    print(f"  Loss: {metrics['loss'][-1]:.6f}")
    print(f"  PSNR: {metrics['psnr'][-1]:.2f} dB")
    print(f"  SSIM: {metrics['ssim'][-1]:.4f}")
    print(f"  Relative Error: {metrics['relative_error'][-1]:.2f}%")
    print(f"Total time: {sum(metrics['time_per_window']):.2f}s\n")
    
    return metrics
