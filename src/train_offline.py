"""Offline (batch) training loop for neural network compression."""

import torch
import torch.nn as nn
import pandas as pd
import os
import json
import time

from .metrics import calculate_metrics_torch, calculate_relative_error


def train_offline(model, train_loader, dataset, device, epochs, model_name, output_dir):
    """
    Train neural network model using offline batch training.

    Args:
        model: Neural network model
        train_loader: PyTorch DataLoader for training data
        dataset: FlowDatasetMinMax instance for denormalization
        device: PyTorch device
        epochs: Number of training epochs
        model_name: Name identifier for saving outputs
        output_dir: Directory path for saving results

    Returns:
        metrics: Dictionary containing training history
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
    print(f"\nOffline training: {model_name}")
    print(f"Training samples: {len(dataset)}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024:.2f} KB\n")

    for epoch in range(epochs):
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

        psnr, ssim = calculate_metrics_torch(all_predictions, all_targets, device)
        rel_error = calculate_relative_error(all_predictions, all_targets)

        metrics['psnr'].append(psnr)
        metrics['ssim'].append(ssim)
        metrics['relative_error'].append(rel_error)

        epoch_time = time.time() - epoch_start
        metrics['time_per_epoch'].append(epoch_time)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.6f}, "
                  f"PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, "
                  f"RE={rel_error:.2f}%, Time={epoch_time:.2f}s")

    model_path = os.path.join(output_dir, f'{model_name}_minmax.pth')
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
    print(f"Final Loss: {metrics['loss'][-1]:.6f}")
    print(f"Final PSNR: {metrics['psnr'][-1]:.2f} dB")
    print(f"Final SSIM: {metrics['ssim'][-1]:.4f}")
    print(f"Final Relative Error: {metrics['relative_error'][-1]:.2f}%")
    print(f"PSNR improvement: {metrics['psnr'][0]:.2f} -> {metrics['psnr'][-1]:.2f} dB "
          f"({metrics['psnr'][-1]-metrics['psnr'][0]:+.2f} dB)\n")

    return metrics


def save_metrics_to_csv(metrics, output_path):
    """
    Save training metrics to CSV file.

    Args:
        metrics: Dictionary containing metric arrays
        output_path: Output file path
    """
    df = pd.DataFrame(metrics)
    df['epoch'] = range(1, len(df) + 1)
    df = df[['epoch', 'loss', 'psnr', 'ssim', 'relative_error', 'time_per_epoch']]
    df.to_csv(output_path, index=False)
    print(f"Metrics saved: {output_path}")
