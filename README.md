# Concurrent Neural Network Training for Compression of Spatio-Temporal Data

**Master's Thesis** | BTU Cottbus-Senftenberg | Mahesh Sadupalli

**[Interactive Demo](https://mahesh-sadupalli.github.io/master-thesis/)** browser-based exploration of flow-field reconstructions, IEEE 754 bit-level compression, and model comparisons.

## What this project does

Modern scientific simulations produce more field data than disks and networks can comfortably move. This project compresses such data by training a small neural network on the simulation output and then storing the trained network in place of the raw arrays. The network is trained while the simulation is still running, so compression happens concurrently with data generation rather than as a post-processing step.

The framework is domain-agnostic and applies to any spatio-temporal field data (CFD, climate modelling, molecular dynamics, structural mechanics). It is validated on a 2D vortex-shedding CFD case study with 7.9 million samples spanning 300 timesteps and four field variables (Vx, Vy, pressure, TKE).

## How it works

Three complementary neural compressors are evaluated, each at three model sizes:

- **Implicit Neural Representation (INR).** A small coordinate-based MLP that maps `(x, y, z, t)` directly to `(Vx, Vy, P, TKE)`. The compressed representation is the network weights alone.
- **Linear Autoencoder.** A fully-connected encoder and decoder that compress each spatial point's temporal sequence into a small latent vector. The compressed representation is the decoder weights plus one latent vector per spatial point.
- **Convolutional Autoencoder.** A 2D-grid encoder and decoder operating on field snapshots that have been interpolated onto a regular grid. The compressed representation is the decoder weights plus one latent vector per timestep.

Each compressor is evaluated in two training modes:

- **Offline (batch).** The model is trained on the full dataset at once; this is the upper bound on reconstruction quality.
- **Online (streaming).** The model is trained sequentially on 15-timestep windows as the simulation produces them. Naive sequential training suffers from catastrophic forgetting; two Experience Replay configurations (ER Scaled and ER Aggressive) mitigate it by replaying past samples from a reservoir-sampled buffer.

## Headline results

| Approach | Best offline PSNR | Best online PSNR | Gap | Best compression |
|----------|------------------:|-----------------:|----:|-----------------:|
| INR | 35.72 dB | 23.27 dB | -12.45 dB | 4,733:1 |
| Linear Autoencoder | 37.94 dB | 34.80 dB | -3.14 dB | 28.6:1 |
| Convolutional Autoencoder | 32.75 dB | 31.05 dB | -1.70 dB | 93.6:1 |

The INR gives the highest compression ratio; the linear autoencoder gives the highest reconstruction quality; the convolutional autoencoder gives the smallest streaming-mode gap, recovering nearly all of its offline quality while still compressing by roughly 94 times.

## Per-approach results (full-dataset PSNR in dB)

### Implicit Neural Representation

| Model | Offline | Naive online | ER Scaled | ER Aggressive |
|-------|--------:|-------------:|----------:|--------------:|
| Base | 31.24 | 14.90 | 21.47 | 21.60 |
| Medium | 34.18 | 13.38 | 21.98 | 23.27 |
| Large | 35.72 | 13.18 | 22.03 | 23.19 |

### Linear Autoencoder

| Model | Offline | Naive online | ER Scaled | ER Aggressive |
|-------|--------:|-------------:|----------:|--------------:|
| Base | 36.23 | 28.51 | 31.38 | 31.18 |
| Medium | 37.90 | 31.16 | 34.45 | 34.46 |
| Large | 37.94 | 30.79 | 34.11 | 34.80 |

### Convolutional Autoencoder

| Model | Offline | Naive online | ER Scaled | ER Aggressive |
|-------|--------:|-------------:|----------:|--------------:|
| Base | 30.67 | 18.53 | 29.18 | 29.32 |
| Medium | 32.75 | 18.35 | 30.48 | 30.93 |
| Large | 32.75 | 18.61 | 30.69 | 31.05 |

## What we learned

- **Catastrophic forgetting is the central challenge** for streaming neural compression. Naive sequential training loses 13 to 21 dB relative to offline training across all three architectures.
- **Experience Replay closes most of the gap.** A reservoir-sampled buffer of past samples, replayed alongside the current window, recovers 6 to 12 dB for the INR and brings the autoencoders to within a few dB of their offline reference.
- **Autoencoders are more streaming-friendly than the INR.** Because they store an explicit per-sample latent code, the shared decoder weights are less prone to interference between temporal windows.
- **Larger INRs forget more, not less.** With shared parameters across the whole domain, additional capacity gives the optimiser more freedom to overwrite earlier mappings during online training.

## Evaluation metrics

- **[PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)** reconstruction quality in dB; higher is better.
- **[SSIM](https://en.wikipedia.org/wiki/Structural_similarity_index_measure)** structural similarity in [0, 1]; higher is better.
- **Relative error** L2 norm error as a percentage of the target norm; lower is better.
- **Compression ratio** original data size divided by the stored representation (model weights, plus latent codes for autoencoders).

## Interactive visualization

A browser-based demo at **[mahesh-sadupalli.github.io/master-thesis](https://mahesh-sadupalli.github.io/master-thesis/)** lets you explore animated flow-field heatmaps, 3D surface views, IEEE 754 bit-level compression effects, and model comparisons across training modes and sizes.

## [License](LICENSE)

MIT
