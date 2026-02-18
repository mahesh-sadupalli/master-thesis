# Concurrent Neural Network Training for Compression of Spatio-Temporal Data

**Master's Thesis** -- M.Sc. Artificial Intelligence

**Author:** Mahesh Sadupalli

**Supervisor:** Prof. Dr.-Ing. Michael Oevermann
**Mentor:** M.Sc. Abhishek Dhiman

**University:** Brandenburgische Technische Universitat Cottbus-Senftenberg (BTU)
**Faculty:** Fachgebiet Numerische Mathematik und Wissenschaftliches Rechnen

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

## Results

### Compression Performance

| Method | Data Size | Base Model CR | Advanced Model CR |
|--------|-----------|---------------|-------------------|
| Binary (.bin) | 253 MB | 8,277 : 1 | 2,379 : 1 |
| CSV raw (.csv) | 833 MB | 27,208 : 1 | 7,819 : 1 |

### Offline vs Online Comparison

| Metric | Base Offline | Base Online | Advanced Offline | Advanced Online |
|--------|-------------|-------------|-----------------|-----------------|
| PSNR (dB) | 30.90 | 11.52 | 35.59 | 6.42 |
| SSIM | 0.920 | 0.758 | 0.980 | 0.684 |
| Rel. Error (%) | 5.29 | -- | 3.04 | -- |

### Key Findings

- **Offline training** achieves excellent reconstruction quality (PSNR > 30 dB, SSIM > 0.92) with extreme compression ratios
- **Online streaming** training suffers from **catastrophic forgetting** -- the model only remembers recent temporal windows, and larger networks are more susceptible
- Online training metrics are misleading: per-window metrics look good, but evaluation on the full dataset reveals significant quality degradation

## Model Architectures

| Model | Architecture | Parameters | Size |
|-------|-------------|------------|------|
| Base | 4 -> 64 -> 64 -> 32 -> 4 | 6,692 | ~30 KB |
| Advanced | 4 -> 128 -> 128 -> 64 -> 4 | 25,668 | ~104 KB |

Both models use ReLU activations, MSE loss, and Adam optimizer (lr=0.001). The network learns a function f(x, y, z, t) -> (Vx, Vy, P, TKE) mapping spatial coordinates and time to flow field variables, treating the data as a continuous implicit neural representation.

## Project Structure

```
master-thesis/
├── src/                    # Source code package
│   ├── models.py           # RegressionModel + AdvancedRegressionModel
│   ├── dataset.py          # FlowDatasetMinMax (min-max normalized)
│   ├── metrics.py          # PSNR, SSIM, relative error calculations
│   ├── train_offline.py    # Offline batch training loop
│   └── train_online.py     # Online streaming training with temporal windows
├── scripts/                # Runnable training & visualization scripts
│   ├── train_base_offline.py
│   ├── visualize_base_offline.py
│   ├── train_advanced_offline.py
│   ├── visualize_advanced_offline.py
│   ├── train_base_online.py
│   ├── visualize_base_online.py
│   ├── train_advanced_online.py
│   └── visualize_advanced_online.py
├── notebooks/              # Jupyter notebooks (all-in-one experiments)
├── data/                   # Dataset directory (CSV not tracked in git)
├── results/                # Training outputs (plots, metrics, JSON)
│   ├── base_model_offline/
│   ├── base_model_online/
│   ├── large_model_offline/
│   └── large_model_online/
└── docs/                   # Thesis document and task description
```

## Setup

```bash
git clone https://github.com/mahesh-sadupalli/master-thesis.git
cd master-thesis

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset

Place the CFD simulation dataset in the `data/` directory:

```
data/ML_test_loader_original_data.csv
```

The dataset is from a vortex shedding CFD simulation containing approximately 7.9 million spatio-temporal samples across 300 timesteps with 8 columns:

- **Inputs (4):** x, y, z, t -- spatial coordinates and time
- **Outputs (4):** Vx, Vy, Pressure, TKE -- velocity components, pressure, and turbulent kinetic energy

## Usage

### Training

Run scripts from the project root:

```bash
# Offline training (batch -- full dataset, multiple epochs)
python scripts/train_base_offline.py
python scripts/train_advanced_offline.py

# Online training (streaming -- temporal windows, incremental)
python scripts/train_base_online.py
python scripts/train_advanced_online.py
```

### Visualization

```bash
# Generate flow field reconstruction plots with error analysis
python scripts/visualize_base_offline.py
python scripts/visualize_advanced_offline.py
python scripts/visualize_base_online.py
python scripts/visualize_advanced_online.py
```

## Evaluation Metrics

- **PSNR (Peak Signal-to-Noise Ratio):** Measures reconstruction quality in dB -- higher is better
- **SSIM (Structural Similarity Index):** Measures structural fidelity between original and reconstructed fields (0 to 1) -- higher is better
- **MSE (Mean Squared Error):** Training loss function measuring average squared reconstruction error
- **Relative Error:** L2 norm error as a percentage of the target norm
- **Compression Ratio:** Original data size divided by model parameter size

## License

MIT
