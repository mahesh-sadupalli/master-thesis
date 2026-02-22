# Concurrent Neural Network Training for Compression of Spatio-Temporal Data

**Master's Thesis** -- M.Sc. Artificial Intelligence

**Author:** Mahesh Sadupalli

## Abstract

This thesis investigates the application of neural networks for concurrent and real-time data compression in streaming spatio-temporal datasets. As modern scientific simulations generate increasingly large data volumes due to higher resolutions and longer runtimes, traditional storage and post-processing approaches face significant I/O bottlenecks and scalability limitations. This work proposes an in-situ and in-transit compression framework that employs deep learning neural networks to learn compact representations of data during runtime.

The methodology integrates neural networks that approximate data patterns as continuous functions of their inputs, replacing large discrete datasets with a compact set of network parameters. This enables concurrent, real-time compression without interrupting primary workflows, reducing the need for storing full data snapshots while maintaining sufficient accuracy for downstream analysis and visualization.

## Problem Statement

Modern scientific simulations generate massive amounts of streaming spatio-temporal data that overwhelm traditional storage and post-processing workflows. Current approaches require storing complete field data at every timestep, incur significant I/O overhead during simulation, and demand large storage requirements for time-resolved datasets. These limitations create bottlenecks that prevent efficient utilization of computational resources and delay scientific insights from simulation data.

**Core question:** *How can we design, implement, and validate a neural network-based compression system that operates concurrently with running simulations, achieves significant data reduction while maintaining scientific accuracy, and integrates seamlessly with existing computational workflows?*

## Research Questions

**RQ1:** How can neural network architectures and training protocols be designed to effectively learn compact representations of streaming spatio-temporal data with limited passes through the dataset?

**RQ2:** How can neural network training and inference be integrated into scientific simulation workflows to enable concurrent, real-time compression without disrupting computational progress or creating I/O bottlenecks?

**RQ3:** What compression performance, reconstruction accuracy, and practical applicability can neural network-based compression achieve compared to traditional methods across diverse spatio-temporal datasets?

## Approach

The core methodology involves training coordinate-based MLPs (Multi-Layer Perceptrons) to learn mappings from spatio-temporal coordinates (x, y, z, t) to flow field variables (Vx, Vy, Pressure, TKE). The network approximates complex flow patterns as continuous functions, effectively replacing large discrete datasets with a compact set of network parameters. Two training paradigms are compared:

- **Offline (batch) training:** The network trains over the entire dataset with multiple epochs, establishing baseline compression performance.
- **Online (streaming) training:** The network trains incrementally using sliding temporal windows, simulating real-time in-situ compression where data arrives sequentially.

## Model Architectures

| Model | Architecture | Parameters | Size |
|-------|-------------|------------|------|
| Base | 4 -> 64 -> 64 -> 32 -> 4 | 6,692 | ~30 KB |
| Medium | 4 -> 96 -> 96 -> 48 -> 4 | 14,644 | ~60 KB |
| Large | 4 -> 128 -> 128 -> 64 -> 4 | 25,668 | ~104 KB |

All models use ReLU activations, MSE loss, and Adam optimizer (lr=0.001). The network learns a function f(x, y, z, t) -> (Vx, Vy, P, TKE) mapping spatial coordinates and time to flow field variables, treating the data as a continuous implicit neural representation.

## Results

### Offline vs Online Comparison

| Metric | Base Offline | Base Online | Medium Offline | Medium Online | Large Offline | Large Online |
|--------|-------------|-------------|----------------|---------------|---------------|--------------|
| PSNR (dB) | 32.15 | 11.97 | 33.58 | 12.70 | 35.99 | 9.67 |
| SSIM | 0.955 | 0.755 | 0.958 | 0.760 | 0.986 | 0.668 |
| Rel. Error (%) | 4.41 | 44.92 | 3.74 | 41.30 | 2.83 | 58.57 |

### Key Findings

- **Offline training** achieves excellent reconstruction quality (PSNR > 32 dB, SSIM > 0.95) with extreme compression ratios across all model sizes
- **Larger models improve offline performance** -- the large model reaches PSNR 35.99 dB and SSIM 0.986, significantly outperforming the base model
- **Online streaming** training suffers from **catastrophic forgetting** -- the model only remembers recent temporal windows, and larger networks are more susceptible
- Online training metrics are misleading: per-window metrics look good, but evaluation on the full dataset reveals significant quality degradation

## Evaluation Metrics

- **PSNR (Peak Signal-to-Noise Ratio):** Measures reconstruction quality in dB -- higher is better
- **SSIM (Structural Similarity Index):** Measures structural fidelity between original and reconstructed fields (0 to 1) -- higher is better
- **MSE (Mean Squared Error):** Training loss function measuring average squared reconstruction error
- **Relative Error:** L2 norm error as a percentage of the target norm
- **Compression Ratio:** Original data size divided by model parameter size

## License

MIT
