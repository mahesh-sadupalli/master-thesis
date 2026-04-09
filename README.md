# Concurrent Neural Network Training for Compression of Spatio-Temporal Data

**Master's Thesis** | BTU Cottbus-Senftenberg | Mahesh Sadupalli

**[Interactive Demo](https://mahesh-sadupalli.github.io/master-thesis/)** -- Explore flow field reconstructions, IEEE 754 bit-level compression visualization, and model comparisons in the browser.

## Abstract

This thesis investigates the application of neural networks for concurrent and real-time data compression in streaming spatio-temporal datasets. As modern scientific simulations generate increasingly large data volumes due to higher resolutions and longer runtimes, traditional storage and post-processing approaches face significant I/O bottlenecks and scalability limitations. This work proposes an in-situ and in-transit compression framework that employs deep learning neural networks to learn compact representations of data during runtime.

The core approach uses **Implicit Neural Representations (INR)** — coordinate-based MLPs that encode entire datasets into compact model weights. Evaluated across three model sizes in two training paradigms (offline batch vs online streaming), with continual learning strategies to mitigate catastrophic forgetting. Validated on a vortex shedding CFD dataset (7.9M samples, 300 timesteps):

- **Batch Learning (Offline)** -- Full-dataset training achieving up to 35.72 dB PSNR with compression ratios exceeding 27,000:1.
- **Continual Learning (Naive Online)** -- Streaming training with temporal windows, revealing catastrophic forgetting as the central challenge (full-dataset PSNR drops to 13-17 dB).
- **Continual Learning with Experience Replay** -- ER-Scaled and ER-Aggressive strategies that mitigate forgetting, improving online PSNR by +6.7-10.1 dB with minimal overhead.

## Approach

The framework is domain-agnostic and applicable to any spatio-temporal field data (CFD, climate modelling, molecular dynamics, structural mechanics, etc.). It maps spatial coordinates and time to field variables using neural networks, replacing large discrete datasets with compact network parameters.

The validation dataset maps `(x, y, z, t) → (Vx, Vy, Pressure, TKE)` from a vortex shedding simulation with 7,919,100 spatio-temporal samples across 300 timesteps.

## Model Architectures

Coordinate-based MLPs (Implicit Neural Representations):

| Model | Architecture | Parameters | Size | Compression Ratio |
|-------|-------------|------------|------|-------------------|
| Base | 4 → 64 → 64 → 32 → 4 | 6,692 | 26.1 KB | 27,395:1 |
| Medium | 4 → 96 → 96 → 48 → 4 | 14,644 | 57.2 KB | 13,241:1 |
| Large | 4 → 128 → 128 → 64 → 4 | 25,668 | 100.3 KB | 7,713:1 |

All models use ReLU activations, MSE loss, and Adam optimizer (lr=0.001).

## Results

### Cross-Method Comparison

| Approach | Model | PSNR (dB) | SSIM | Rel. Error (%) | Compression | Training Time |
|----------|-------|-----------|------|----------------|-------------|---------------|
| **Batch Learning** | Base | 31.24 | 0.9748 | 4.90 | 27,395:1 | ~3.2 hrs |
| | Medium | 34.18 | 0.9853 | 3.49 | 13,241:1 | ~3.2 hrs |
| | Large | 35.72 | 0.9823 | 2.92 | 7,713:1 | ~3.2 hrs |
| **Naive Online** | Base | 14.90 | 0.7996 | 32.07 | 27,395:1 | 42.7s |
| | Medium | 13.38 | 0.7822 | 38.21 | 13,241:1 | 71.6s |
| | Large | 13.18 | 0.7941 | 39.07 | 7,713:1 | 85.3s |
| **ER Scaled** | Base | 21.47 | 0.8830 | 15.07 | 27,395:1 | 49.9s |
| | Medium | 21.98 | 0.8782 | 14.22 | 13,241:1 | 77.4s |
| | Large | 22.03 | 0.9034 | 14.14 | 7,713:1 | 90.7s |
| **ER Aggressive** | Base | 21.60 | 0.8958 | 14.86 | 27,395:1 | 59.4s |
| | Medium | 23.27 | 0.8961 | 12.26 | 13,241:1 | 88.2s |
| | Large | 23.19 | 0.8931 | 12.37 | 7,713:1 | 101.3s |

### Flow Field Reconstructions -- Batch Learning (Offline)

| Base (31.24 dB) | Medium (34.18 dB) | Large (35.72 dB) |
|:---:|:---:|:---:|
| ![](results/batch_learning/base_model_offline/base_flow_visualization.png) | ![](results/batch_learning/medium_model_offline/medium_flow_visualization.png) | ![](results/batch_learning/large_model_offline/large_flow_visualization.png) |

### Flow Field Reconstructions -- Continual Learning (Naive Online)

Full-dataset evaluation reveals severe quality degradation due to catastrophic forgetting.

| Base (14.90 dB) | Medium (13.38 dB) | Large (13.18 dB) |
|:---:|:---:|:---:|
| ![](results/continual_learning/base_model_online/base_online_visualization.png) | ![](results/continual_learning/medium_model_online/medium_online_visualization.png) | ![](results/continual_learning/large_model_online/large_online_visualization.png) |

### Flow Field Reconstructions -- Experience Replay

| Naive | ER Scaled | ER Aggressive |
|:---:|:---:|:---:|
| ![](results/cl_boosting/naive_flow_field.png) | ![](results/cl_boosting/er_scaled_flow_field.png) | ![](results/cl_boosting/er_aggressive_flow_field.png) |

| PSNR per Window | Gap to Offline |
|:---:|:---:|
| ![](results/cl_boosting/comparison_cl/cl_psnr_per_window.png) | ![](results/cl_boosting/comparison_cl/cl_gap_to_offline.png) |

## Key Findings

1. **Batch learning achieves excellent reconstruction** -- up to 35.72 dB PSNR and 0.982 SSIM with extreme compression ratios (27,395:1) but requires hours of training
2. **Catastrophic forgetting is severe in naive online training** -- larger models forget more (counter-intuitive), with full-dataset PSNR dropping to 13-15 dB
3. **Experience Replay effectively mitigates forgetting** -- ER-Aggressive improves online PSNR by +6.7-10.1 dB with only 14-40% computational overhead
4. **Regularization methods (EWC, LwF) fail for INR compression** -- shared parameter spaces make task-specific protection impossible; LwF is actively harmful for large models
5. **10-12 dB gap to offline remains** -- motivating future work with autoencoder architectures and larger replay buffers

## Evaluation Metrics

- **[PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) (Peak Signal-to-Noise Ratio):** Reconstruction quality in dB -- higher is better
- **[SSIM](https://en.wikipedia.org/wiki/Structural_similarity_index_measure) (Structural Similarity Index):** Structural fidelity (0 to 1) -- higher is better
- **Relative Error:** L2 norm error as a percentage of the target norm
- **Compression Ratio:** Original data size divided by model parameter size

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
