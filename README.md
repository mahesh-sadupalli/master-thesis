# Neural Network-Based Spatio-Temporal Flow Field Data Compression

Master's thesis project comparing **offline (batch)** vs **online (streaming)** training approaches for neural network-based compression of spatio-temporal flow field simulation data.

## Key Findings

| Metric | Base Offline | Base Online | Advanced Offline | Advanced Online |
|--------|-------------|-------------|-----------------|-----------------|
| PSNR (dB) | 30.90 | 11.52 | 35.59 | 6.42 |
| SSIM | 0.920 | 0.758 | 0.980 | 0.684 |
| Parameters | 6,692 | 6,692 | 25,668 | 25,668 |

- Offline training achieves compression ratios up to **8,277:1** (base) and **2,379:1** (advanced)
- Online streaming training suffers from **catastrophic forgetting** -- larger models are more susceptible

## Project Structure

```
master-thesis/
├── src/                    # Source code package
│   ├── models.py           # RegressionModel + AdvancedRegressionModel
│   ├── dataset.py          # FlowDatasetMinMax (min-max normalized)
│   ├── metrics.py          # PSNR, SSIM, relative error
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
│   ├── visualize_advanced_online.py
│   └── generate_pptx.py
├── notebooks/              # Jupyter notebooks (all-in-one experiments)
├── data/                   # Dataset directory (CSV not tracked)
├── results/                # Training outputs (plots, metrics, JSON)
└── docs/                   # Documentation
```

## Setup

```bash
git clone https://github.com/<your-username>/master-thesis.git
cd master-thesis

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data

Place the dataset CSV in the `data/` directory:
```
data/ML_test_loader_original_data.csv
```

The dataset contains 7,919,100 rows with 8 columns:
- **Inputs (4):** x, y, z, t (spatial coordinates + time)
- **Outputs (4):** Vx, Vy, Pressure, TKE (flow field variables)

## Usage

### Training

Run scripts from the project root:

```bash
# Offline training
python scripts/train_base_offline.py
python scripts/train_advanced_offline.py

# Online training
python scripts/train_base_online.py
python scripts/train_advanced_online.py
```

### Visualization

```bash
python scripts/visualize_base_offline.py
python scripts/visualize_advanced_offline.py
python scripts/visualize_base_online.py
python scripts/visualize_advanced_online.py
```

### Generate Comparison Presentation

```bash
python scripts/generate_pptx.py
```

## Model Architectures

- **Base Model:** 4 -> 64 -> 64 -> 32 -> 4 (6,692 parameters, ~30 KB)
- **Advanced Model:** 4 -> 128 -> 128 -> 64 -> 4 (25,668 parameters, ~104 KB)

Both use ReLU activations and MSE loss with Adam optimizer (lr=0.001).

## License

MIT
