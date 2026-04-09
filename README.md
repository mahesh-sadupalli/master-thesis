# Concurrent Neural Network Training for Compression of Spatio-Temporal Data

**Master's Thesis** | BTU Cottbus-Senftenberg | Mahesh Sadupalli

**[Interactive Demo](https://mahesh-sadupalli.github.io/master-thesis/)** -- Explore flow field reconstructions, IEEE 754 bit-level compression visualization, and model comparisons in the browser.

## Abstract

This thesis investigates the application of neural networks for concurrent and real-time data compression in streaming spatio-temporal datasets. As modern scientific simulations generate increasingly large data volumes due to higher resolutions and longer runtimes, traditional storage and post-processing approaches face significant I/O bottlenecks and scalability limitations. This work proposes an in-situ and in-transit compression framework that employs deep learning neural networks to learn compact representations of data during runtime.

Four compression architectures were systematically evaluated across two training paradigms (offline batch vs online streaming) with three model sizes each, plus continual learning strategies (Naive, ER Scaled, ER Aggressive) to mitigate catastrophic forgetting in online mode. Validated on a vortex shedding CFD dataset (7.9M samples, 300 timesteps):

1. **Implicit Neural Representations (INR)** -- Coordinate-based MLPs encoding the entire dataset into model weights: up to 35.72 dB PSNR with 27,395:1 compression (offline); catastrophic forgetting in online mode mitigated by Experience Replay (+8.6 dB).
2. **Linear Autoencoder** -- Temporal-point encoding compressing per-point time series: best reconstruction quality at 37.94 dB PSNR (offline); online + ER Aggressive achieves 34.80 dB (gap of only -3.14 dB to offline).
3. **Convolutional Autoencoder** -- Grid-based Conv2D compression: best compression efficiency at 93.6:1 with 30.67 dB PSNR (offline).

Autoencoders proved significantly more streaming-friendly than INRs, with Linear AE showing -3.14 dB offline-to-online gap compared to INR (-12.5 dB).

## Approach

The framework is domain-agnostic and applicable to any spatio-temporal field data (CFD, climate modelling, molecular dynamics, structural mechanics, etc.). It maps spatial coordinates and time to field variables using neural networks, replacing large discrete datasets with compact network parameters.

The validation dataset maps `(x, y, z, t) → (Vx, Vy, Pressure, TKE)` from a vortex shedding simulation with 7,919,100 spatio-temporal samples across 300 timesteps.

## Model Architectures

### Implicit Neural Representations (INR)

Coordinate-based MLPs used for batch learning, continual learning, and experience replay approaches:

| Model | Architecture | Parameters | Size |
|-------|-------------|------------|------|
| Base | 4 → 64 → 64 → 32 → 4 | 6,692 | 26.1 KB |
| Medium | 4 → 96 → 96 → 48 → 4 | 14,644 | 57.2 KB |
| Large | 4 → 128 → 128 → 64 → 4 | 25,668 | 100.3 KB |

### Linear Autoencoder

Fully connected autoencoders compressing per-point temporal windows (60-dim → latent). Input: 15 timesteps × 4 variables = 60 per spatial point:

| Model | Architecture | Latent Dim | Parameters | Compression |
|-------|-------------|-----------|------------|-------------|
| Base | 60 → 64 → 32 → 8 → 32 → 64 → 60 | 8 | 12,548 | 28.6:1 |
| Medium | 60 → 128 → 64 → 16 → 64 → 128 → 60 | 16 | 34,252 | 13.1:1 |
| Large | 60 → 256 → 128 → 32 → 128 → 256 → 60 | 32 | 117,724 | 9.7:1 |

### Convolutional Autoencoder

Conv2D autoencoders operating on 32×128 interpolated field grids:

| Model | Channels | Latent Dim | Parameters | Total Size |
|-------|----------|-----------|------------|------------|
| Base | 4 → 16 → 32 → 64 → 128 | 32 | 328,900 | 1,322.3 KB |
| Medium | 4 → 32 → 64 → 128 → 256 | 64 | 1,307,012 | 5,180.5 KB |
| Large | 4 → 32 → 64 → 128 → 256 | 128 | 1,831,364 | 7,303.8 KB |

## Results

### Cross-Method Comparison

| Approach | Model | PSNR (dB) | SSIM | Rel. Error (%) | Compression | Training Time |
|----------|-------|-----------|------|----------------|-------------|---------------|
| **Batch Learning** | Base | 31.24 | 0.9748 | 4.90 | 27,395:1 | ~3.2 hrs |
| | Medium | 34.18 | 0.9853 | 3.49 | 13,241:1 | ~3.2 hrs |
| | Large | 35.72 | 0.9823 | 2.92 | 7,713:1 | ~3.2 hrs |
| **Naive Online** | Base | 17.32 | 0.8470 | 24.28 | 27,395:1 | 39.8s |
| | Medium | 15.21 | 0.8083 | 30.96 | 13,241:1 | 61.0s |
| | Large | 14.45 | 0.7926 | 33.75 | 7,713:1 | 84.6s |
| **ER Scaled** | Base | 21.47 | 0.8830 | 15.07 | 27,395:1 | 49.9s |
| | Medium | 21.98 | 0.8782 | 14.22 | 13,241:1 | 77.4s |
| | Large | 22.03 | 0.9034 | 14.14 | 7,713:1 | 90.7s |
| **ER Aggressive** | Base | 21.60 | 0.8958 | 14.86 | 27,395:1 | 59.4s |
| | Medium | 23.27 | 0.8961 | 12.26 | 13,241:1 | 88.2s |
| | Large | 23.19 | 0.8931 | 12.37 | 7,713:1 | 101.3s |
| **Linear AE** | Base | 36.23 | 0.9697 | 2.74 | 28.6:1 | 57.3s |
| | Medium | 37.90 | 0.9719 | 2.26 | 13.1:1 | 62.0s |
| | Large | 37.94 | 0.9749 | 2.25 | 9.7:1 | 61.9s |
| **Linear AE Naive** | Base | 28.51 | 0.9751 | 6.71 | 28.6:1 | — |
| | Medium | 31.16 | 0.9769 | 4.95 | 13.1:1 | — |
| | Large | 30.79 | 0.9805 | 5.17 | 9.7:1 | — |
| **Linear AE ER Scaled** | Base | 31.38 | 0.9769 | 4.82 | 28.6:1 | — |
| | Medium | 34.45 | 0.9789 | 3.39 | 13.1:1 | — |
| | Large | 34.11 | 0.8850 | 3.52 | 9.7:1 | — |
| **Linear AE ER Aggressive** | Base | 31.18 | 0.9750 | 4.94 | 28.6:1 | — |
| | Medium | 34.46 | 0.9865 | 3.38 | 13.1:1 | — |
| | Large | 34.80 | 0.9859 | 3.26 | 9.7:1 | — |
| **Conv AE** | Base | 30.67 | 0.9574 | 5.22 | 93.6:1 | 19.9s |
| | Medium | 32.75 | 0.9723 | 4.10 | 23.9:1 | 16.4s |
| | Large | 32.75 | 0.9704 | 4.10 | 16.9:1 | 16.6s |

### Comparison Across All Methods

| PSNR | SSIM |
|:---:|:---:|
| ![](results/comparison_all_methods/psnr.png) | ![](results/comparison_all_methods/ssim.png) |

| Loss | Relative Error |
|:---:|:---:|
| ![](results/comparison_all_methods/loss.png) | ![](results/comparison_all_methods/relative_error.png) |

### Flow Field Reconstructions -- Batch Learning (Offline)

| Base (31.24 dB) | Medium (34.18 dB) | Large (35.72 dB) |
|:---:|:---:|:---:|
| ![](results/batch_learning/base_model_offline/base_flow_visualization.png) | ![](results/batch_learning/medium_model_offline/medium_flow_visualization.png) | ![](results/batch_learning/large_model_offline/large_flow_visualization.png) |

### Flow Field Reconstructions -- Continual Learning (Naive Online)

Full-dataset evaluation reveals severe quality degradation due to catastrophic forgetting.

| Base (17.32 dB) | Medium (15.21 dB) | Large (14.45 dB) |
|:---:|:---:|:---:|
| ![](results/continual_learning/base_model_online/base_online_visualization.png) | ![](results/continual_learning/medium_model_online/medium_online_visualization.png) | ![](results/continual_learning/large_model_online/large_online_visualization.png) |

### Flow Field Reconstructions -- Experience Replay

| Naive | ER Scaled | ER Aggressive |
|:---:|:---:|:---:|
| ![](results/cl_boosting/naive_flow_field.png) | ![](results/cl_boosting/er_scaled_flow_field.png) | ![](results/cl_boosting/er_aggressive_flow_field.png) |

| PSNR per Window | Gap to Offline |
|:---:|:---:|
| ![](results/cl_boosting/comparison_cl/cl_psnr_per_window.png) | ![](results/cl_boosting/comparison_cl/cl_gap_to_offline.png) |

### Flow Field Reconstructions -- Linear Autoencoder (Offline)

| Base (36.23 dB) | Medium (37.90 dB) | Large (37.94 dB) |
|:---:|:---:|:---:|
| ![](results/autoencoder/linear_ae/base_autoencoder/base_ae_flow_visualization.png) | ![](results/autoencoder/linear_ae/medium_autoencoder/medium_ae_flow_visualization.png) | ![](results/autoencoder/linear_ae/large_autoencoder/large_ae_flow_visualization.png) |

### Flow Field Reconstructions -- Linear Autoencoder (Online + CL)

Best online: Large ER Aggressive at 34.80 dB (gap of only -3.14 dB to offline).

| Naive | ER Scaled | ER Aggressive |
|:---:|:---:|:---:|
| ![](results/autoencoder/linear_ae_online/ae_online_results/ae_naive_flow_field.png) | ![](results/autoencoder/linear_ae_online/ae_online_results/ae_er_scaled_flow_field.png) | ![](results/autoencoder/linear_ae_online/ae_online_results/ae_er_aggressive_flow_field.png) |

| PSNR per Window | Gap to Offline |
|:---:|:---:|
| ![](results/autoencoder/linear_ae_online/ae_online_results/comparison_ae_online/ae_online_psnr_per_window.png) | ![](results/autoencoder/linear_ae_online/ae_online_results/comparison_ae_online/ae_online_gap_to_offline.png) |

### Flow Field Reconstructions -- Convolutional Autoencoder (Offline)

| Base (30.67 dB) | Medium (32.75 dB) | Large (32.75 dB) |
|:---:|:---:|:---:|
| ![](results/autoencoder/conv2d/base/base_conv_ae_flow_visualization.png) | ![](results/autoencoder/conv2d/medium/medium_conv_ae_flow_visualization.png) | ![](results/autoencoder/conv2d/large/large_conv_ae_flow_visualization.png) |


## Key Findings

1. **Linear Autoencoder achieves the best reconstruction quality** -- 37.94 dB PSNR offline; online + ER Aggressive reaches 34.80 dB (gap of only -3.14 dB)
2. **Convolutional Autoencoder offers the best compression efficiency** -- 93.6:1 compression ratio with 30.67 dB PSNR and only 20 seconds of training
3. **Autoencoders are far more streaming-friendly than INRs** -- Linear AE gap to offline is -3.14 dB vs -12.5 dB for INR, making AE architectures practical for real-time compression
4. **Batch learning INR provides extreme compression** -- up to 27,395:1 ratio with 35.72 dB, but requires hours of training on the full dataset
5. **Catastrophic forgetting is severe in naive online training** -- larger models forget more (counter-intuitive), with INR PSNR dropping to 13-17 dB
6. **Experience Replay is the most effective CL strategy** -- consistently improves all architectures; ER-Aggressive best for quality, ER-Scaled best for efficiency
7. **Regularization methods (EWC, LwF) fail for INR compression** -- shared parameter spaces make task-specific protection impossible; LwF is actively harmful for large models
8. **Each approach occupies a distinct point in the accuracy-compression-speed trade-off space** -- no single method dominates across all criteria

## Evaluation Metrics

- **[PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) (Peak Signal-to-Noise Ratio):** Reconstruction quality in dB -- higher is better
- **[SSIM](https://en.wikipedia.org/wiki/Structural_similarity_index_measure) (Structural Similarity Index):** Structural fidelity (0 to 1) -- higher is better
- **Relative Error:** L2 norm error as a percentage of the target norm
- **Compression Ratio:** Original data size divided by total stored size (model weights + latent codes)

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
