"""
Regularization-based: Elastic Weight Consolidation (EWC).

Penalizes changes to parameters deemed important for previous windows using
the diagonal of the Fisher Information Matrix (Kirkpatrick et al., 2017).
"""

import sys
import os
import torch
import pandas as pd
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from unified_training_utils import (
    SpatioTemporalDataset, BaseCompressor, MediumCompressor, LargeCompressor
)
from continual_learning.cl_strategies import EWCStrategy
from continual_learning.cl_training import train_online_cl, evaluate_full_dataset
from continual_learning.experiments.config import (
    DATA_FILE, RESULTS_BASE, get_device, EPOCHS_PER_WINDOW, NUM_WINDOWS,
    EWC_DEFAULTS, OFFLINE_REFERENCE
)

MODELS = {
    "base": BaseCompressor,
    "medium": MediumCompressor,
    "large": LargeCompressor,
}


def run(model_name="base"):
    """Run Elastic Weight Consolidation experiment for a single model size."""
    device = get_device()
    print(f"[Environment] Device: {device}")

    print(f"[Data] Loading dataset: {DATA_FILE}")
    dataset = SpatioTemporalDataset(DATA_FILE)
    print(f"[Data] Loaded {len(dataset.inputs)} samples")

    model = MODELS[model_name]().to(device)
    strategy = EWCStrategy(**EWC_DEFAULTS)
    output_dir = os.path.join(RESULTS_BASE, f"{model_name}_ewc")

    # Train
    metrics = train_online_cl(
        model=model, dataset=dataset, device=device,
        epochs_per_window=EPOCHS_PER_WINDOW,
        model_name=f"{model_name}_ewc",
        output_dir=output_dir, strategy=strategy,
        num_windows=NUM_WINDOWS,
    )

    # Save per-window metrics
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(output_dir, f"{model_name}_ewc_metrics.csv"), index=False)

    # Full dataset evaluation
    full_eval = evaluate_full_dataset(
        model, dataset, device, model_name=f"{model_name}_ewc"
    )

    # Save evaluation
    eval_data = {
        "model": f"{model_name}_ewc",
        "training_mode": "online_streaming",
        "strategy": strategy.get_config(),
        "parameters": sum(p.numel() for p in model.parameters()),
        **full_eval,
    }
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(eval_data, f, indent=2)

    # Forgetting gap relative to offline
    ref = OFFLINE_REFERENCE[model_name]
    print(f"[Comparison] Offline reference PSNR: {ref['psnr_db']:.2f} dB")
    print(f"[Comparison] Gap to offline: {full_eval['psnr_db'] - ref['psnr_db']:.2f} dB")

    return {
        "model": model_name,
        "last_window_psnr": metrics["psnr"][-1],
        "full_dataset_psnr": full_eval["psnr_db"],
        "full_dataset_ssim": full_eval["ssim"],
        "full_dataset_re": full_eval["relative_error_pct"],
        "psnr_drop": metrics["psnr"][-1] - full_eval["psnr_db"],
        "total_time": sum(metrics["time_per_window"]),
    }


def run_all():
    """Run Elastic Weight Consolidation experiment across all model sizes."""
    results = []
    for model_name in MODELS:
        result = run(model_name)
        results.append(result)

    print(f"\n[Summary] Elastic Weight Consolidation (lambda={EWC_DEFAULTS['ewc_lambda']}, "
          f"fisher_samples={EWC_DEFAULTS['fisher_samples']})")
    print(f"{'Model':<10} {'Window PSNR':>12} {'Full PSNR':>10} {'Drop':>8} {'SSIM':>8} {'RE%':>8} {'Time':>8}")
    print("-" * 68)
    for r in results:
        print(f"{r['model']:<10} {r['last_window_psnr']:>10.2f} dB "
              f"{r['full_dataset_psnr']:>8.2f} dB {r['psnr_drop']:>6.2f} dB "
              f"{r['full_dataset_ssim']:>8.4f} {r['full_dataset_re']:>7.2f}% "
              f"{r['total_time']:>7.1f}s")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Elastic Weight Consolidation (EWC)"
    )
    parser.add_argument("--model", type=str, default=None,
                        choices=["base", "medium", "large"],
                        help="Run single model (default: all)")
    args = parser.parse_args()

    if args.model:
        run(args.model)
    else:
        run_all()
