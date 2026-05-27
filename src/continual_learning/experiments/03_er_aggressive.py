"""
Online INR with Experience Replay (Aggressive configuration).

Trains the coordinate-based INR sequentially across 20 temporal windows
with a replay buffer of 100,000 samples and replay weight 1.0.
"""

import sys
import os
import pandas as pd
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from unified_training_utils import (
    SpatioTemporalDataset, BaseCompressor, MediumCompressor, LargeCompressor
)
from continual_learning.cl_strategies import ExperienceReplayStrategy
from continual_learning.cl_training import train_online_cl, evaluate_full_dataset
from continual_learning.experiments.config import (
    DATA_FILE, RESULTS_BASE, get_device, EPOCHS_PER_WINDOW, NUM_WINDOWS,
    ER_AGGRESSIVE_DEFAULTS, OFFLINE_REFERENCE
)

MODELS = {
    "base": BaseCompressor,
    "medium": MediumCompressor,
    "large": LargeCompressor,
}


def run(model_name="base"):
    """Run Aggressive Experience Replay experiment for a single model size."""
    device = get_device()
    print("[Environment] Device: {}".format(device))

    print("[Data] Loading dataset: {}".format(DATA_FILE))
    dataset = SpatioTemporalDataset(DATA_FILE)
    print("[Data] Loaded {} samples".format(len(dataset.inputs)))

    model = MODELS[model_name]().to(device)
    strategy = ExperienceReplayStrategy(**ER_AGGRESSIVE_DEFAULTS)
    output_dir = os.path.join(RESULTS_BASE, "{}_er_aggressive".format(model_name))

    metrics = train_online_cl(
        model=model, dataset=dataset, device=device,
        epochs_per_window=EPOCHS_PER_WINDOW,
        model_name="{}_er_aggressive".format(model_name),
        output_dir=output_dir, strategy=strategy,
        num_windows=NUM_WINDOWS,
    )

    df = pd.DataFrame(metrics)
    df.to_csv(
        os.path.join(output_dir, "{}_er_aggressive_metrics.csv".format(model_name)),
        index=False,
    )

    full_eval = evaluate_full_dataset(
        model, dataset, device, model_name="{}_er_aggressive".format(model_name)
    )

    eval_data = {
        "model": "{}_er_aggressive".format(model_name),
        "training_mode": "online_streaming",
        "strategy": strategy.get_config(),
        "parameters": sum(p.numel() for p in model.parameters()),
        **full_eval,
    }
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(eval_data, f, indent=2)

    ref = OFFLINE_REFERENCE[model_name]
    print("[Comparison] Offline reference PSNR: {:.2f} dB".format(ref["psnr_db"]))
    print("[Comparison] Gap to offline: {:.2f} dB".format(
        full_eval["psnr_db"] - ref["psnr_db"]
    ))

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
    """Run Aggressive Experience Replay experiment across all model sizes."""
    results = []
    for model_name in MODELS:
        result = run(model_name)
        results.append(result)

    print("\n[Summary] Aggressive Experience Replay (buffer={}, weight={})".format(
        ER_AGGRESSIVE_DEFAULTS["buffer_size"],
        ER_AGGRESSIVE_DEFAULTS["replay_weight"],
    ))
    print("{:<10} {:>12} {:>10} {:>8} {:>8} {:>8} {:>8}".format(
        "Model", "Window PSNR", "Full PSNR", "Drop", "SSIM", "RE%", "Time"
    ))
    print("-" * 68)
    for r in results:
        print("{:<10} {:>10.2f} dB {:>8.2f} dB {:>6.2f} dB {:>8.4f} {:>7.2f}% {:>7.1f}s".format(
            r["model"], r["last_window_psnr"], r["full_dataset_psnr"],
            r["psnr_drop"], r["full_dataset_ssim"], r["full_dataset_re"],
            r["total_time"],
        ))

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Online INR with Aggressive Experience Replay"
    )
    parser.add_argument("--model", type=str, default=None,
                        choices=["base", "medium", "large"],
                        help="Run single model (default: all)")
    args = parser.parse_args()

    if args.model:
        run(args.model)
    else:
        run_all()
