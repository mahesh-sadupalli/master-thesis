# Concurrent Neural Network Training for Compression of Spatio-Temporal Data

**Master's Thesis** | BTU Cottbus-Senftenberg | Mahesh Sadupalli

**[Interactive Demo](https://mahesh-sadupalli.github.io/master-thesis/)** Explore flow field reconstructions, IEEE 754 bit-level compression visualization, and model comparisons in the browser.

## Abstract

This thesis investigates the application of neural networks for concurrent and real-time data compression in streaming spatio-temporal datasets. As modern scientific simulations generate increasingly large data volumes due to higher resolutions and longer runtimes, traditional storage and post-processing approaches face significant I/O bottlenecks and scalability limitations. This work proposes an in-situ and in-transit compression framework that employs deep learning neural networks to learn compact representations of data during runtime.

Three complementary compression approaches are developed and evaluated under both offline batch training and online streaming training, with continual learning strategies to mitigate catastrophic forgetting:

- **Implicit Neural Representations (INR):** coordinate-based MLPs that encode entire datasets into compact model weights.
- **Linear Autoencoder (LAE):** fully-connected encoder/decoder operating on per-point temporal sequences.
- **Convolutional Autoencoder (Conv2D AE):** 2D-grid based encoder/decoder operating on field snapshots after Delaunay interpolation onto a regular grid.

The framework is validated on a vortex shedding CFD dataset (7.9M samples, 300 timesteps) chosen as a representative case study; the methodology itself is domain-agnostic and applies to any spatio-temporal field data.

## Approach

The framework maps spatial coordinates and time to field variables using neural networks, replacing large discrete datasets with compact network parameters and (for autoencoders) per-sample latent codes.

The validation dataset maps `(x, y, z, t) -> (Vx, Vy, Pressure, TKE)` from a vortex shedding simulation with 7,919,100 spatio-temporal samples across 300 timesteps.

## Model Architectures

### Implicit Neural Representation (Coordinate-Based MLP)

| Model | Architecture | Parameters | Size | Compression Ratio |
|-------|-------------|------------|------|-------------------|
| Base | 4 -> 64 -> 64 -> 32 -> 4 | 6,692 | 29 KB | 4,733:1 |
| Medium | 4 -> 96 -> 96 -> 48 -> 4 | 14,644 | 60 KB | 2,163:1 |
| Large | 4 -> 128 -> 128 -> 64 -> 4 | 25,668 | 103 KB | 1,234:1 |

### Linear Autoencoder (Offline)

| Model | Architecture | Latent Dim | Parameters | Compression Ratio |
|-------|-------------|------------|------------|-------------------|
| Base | 1200 -> 256 -> 128 -> 16 -> 128 -> 256 -> 1200 | 16 | 686,016 | 28.6:1 |
| Medium | 1200 -> 512 -> 256 -> 128 -> 32 -> 128 -> 256 -> 512 -> 1200 | 32 | 1,567,696 | 13.1:1 |
| Large | 1200 -> 512 -> 256 -> 128 -> 64 -> 128 -> 256 -> 512 -> 1200 | 64 | 1,575,920 | 9.7:1 |

### Convolutional Autoencoder (2D-grid)

| Model | Encoder Channels | Latent Dim | Parameters | Compression Ratio |
|-------|------------------|------------|------------|-------------------|
| Base | 4 -> 16 -> 32 -> 64 -> 128 | 32 | 328,900 | 93.6:1 |
| Medium | 4 -> 32 -> 64 -> 128 -> 256 | 64 | 1,307,012 | 23.9:1 |
| Large | 4 -> 32 -> 64 -> 128 -> 256 | 128 | 1,831,364 | 16.9:1 |

All models are trained with the Adam optimiser (lr = 0.001) and MSE loss. INR uses ReLU activations on hidden layers; the linear autoencoder uses Leaky ReLU with dropout; the convolutional autoencoder uses Leaky ReLU with batch normalisation.

## Results (Full-Dataset PSNR in dB)

### Cross-Method Summary (best result per approach)

| Approach | Best PSNR (dB) | Gap to Offline | Best CR |
|----------|---------------|----------------|---------|
| Batch INR (Offline) | 35.72 | reference | 4,733:1 |
| Online INR + CL | 23.27 | -12.45 dB | 4,733:1 |
| Linear AE (Offline) | 37.94 | reference | 28.6:1 |
| Online LAE + CL | 34.80 | -3.14 dB | 9.7:1 |
| Conv2D AE (Offline) | 32.75 | reference | 93.6:1 |
| Online Conv2D + CL | 31.05 | -1.70 dB | 93.6:1 |

### INR: Offline vs Online (Full-Dataset)

| Model | Offline | Naive | ER Scaled | ER Aggressive |
|-------|---------|-------|-----------|---------------|
| Base | 31.24 | 14.90 | 21.47 | 21.60 |
| Medium | 34.18 | 13.38 | 21.98 | 23.27 |
| Large | 35.72 | 13.18 | 22.03 | 23.19 |

### Linear Autoencoder: Offline vs Online (Full-Dataset)

| Model | Offline | Naive | ER Scaled | ER Aggressive |
|-------|---------|-------|-----------|---------------|
| Base | 36.23 | 28.51 | 31.38 | 31.18 |
| Medium | 37.90 | 31.16 | 34.45 | 34.46 |
| Large | 37.94 | 30.79 | 34.11 | 34.80 |

### Convolutional Autoencoder: Offline vs Online (Full-Dataset)

| Model | Offline | Naive | ER Scaled | ER Aggressive |
|-------|---------|-------|-----------|---------------|
| Base | 30.67 | 18.53 | 29.18 | 29.32 |
| Medium | 32.75 | 18.35 | 30.48 | 30.93 |
| Large | 32.75 | 18.61 | 30.69 | 31.05 |

## Key Findings

1. **All three approaches achieve high-quality offline reconstruction.** The linear autoencoder reaches the highest reconstruction quality (37.94 dB), the INR achieves the highest compression ratios (up to 4,733:1), and the convolutional autoencoder balances both (32.75 dB at 93.6:1).
2. **Naive online training suffers severe catastrophic forgetting** for all architectures, with full-dataset PSNR dropping by 13 to 21 dB relative to offline.
3. **Experience Replay substantially mitigates forgetting.** ER Aggressive recovers 6 to 12 dB for the INR, 3 to 4 dB for the linear autoencoder, and 11 to 12 dB for the convolutional autoencoder.
4. **Autoencoders are more streaming-friendly than the INR.** The Conv2D AE with ER Aggressive closes the gap to within 1.70 dB of offline; the linear AE to within 3.14 dB; the INR retains an approximately 12 dB residual gap.
5. **Larger INRs forget more severely** than smaller ones under naive online training, an inversion of the offline scaling trend, indicating that the failure mode is interference among shared parameters rather than insufficient capacity.

## Evaluation Metrics

- **[PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) (Peak Signal-to-Noise Ratio):** Reconstruction quality in dB; higher is better.
- **[SSIM](https://en.wikipedia.org/wiki/Structural_similarity_index_measure) (Structural Similarity Index):** Structural fidelity in [0, 1]; higher is better.
- **Relative Error:** L2 norm error as a percentage of the target norm.
- **Compression Ratio:** Original data size divided by stored representation (model weights, plus latent codes for autoencoders).

## Repository Structure

```
src/
  unified_training_utils.py        INR dataset, models, metrics, offline training loop
  autoencoder_utils.py             Linear autoencoder model and training utilities
  online_training_utils.py         Online window-based training loop for the INR
  train_{base,medium,large}_{offline,online}.py   INR training entry points
  visualize_*.py                   Flow-field visualisation scripts
  compare_{offline,online}.py      Cross-model comparison plots
  regen_time_avg_multistation.py   Time-averaged wake-profile figure
  continual_learning/
    cl_strategies.py               Naive and Experience Replay strategies
    cl_training.py                 Online CL training loop for the INR
    replay_buffer.py               Reservoir-sampling replay buffer
    ae_cl_training.py              Online CL training loop for the linear AE
    ae_online_dataset.py           WindowedAEDataset for online AE training
    experiments/                   Per-strategy experiment entry points
notebooks/                         Kaggle-ready notebooks for all approaches
results/                           Trained model checkpoints and metrics
documents/                         Thesis LaTeX sources and PDF
```

## Interactive Visualization

An interactive browser-based demo is available at **[mahesh-sadupalli.github.io/master-thesis](https://mahesh-sadupalli.github.io/master-thesis/)**, featuring:

- Animated 2D flow field heatmaps (Original, Predicted, Absolute Error)
- 3D surface visualization with orbit controls
- IEEE 754 float64 bit-level comparison showing compression effects at individual grid points
- Coordinate system scatter plots showing data distribution
- Model comparison across training modes and model sizes
- Real-time metrics display (PSNR, SSIM, Relative Error, Compression Ratio)

## [License](LICENSE)

MIT
