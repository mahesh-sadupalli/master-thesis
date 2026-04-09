"""
Online Linear Autoencoder — Scaled Experience Replay

Uses larger buffer (50K) and higher replay weight (0.7) to mitigate
catastrophic forgetting during online AE training.
"""

import sys
import os
import pandas as pd
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from autoencoder_utils import LinearAutoEncoder
from continual_learning.cl_strategies import ExperienceReplayStrategy
from continual_learning.ae_online_dataset import WindowedAEDataset
from continual_learning.ae_cl_training import train_online_ae_cl, evaluate_full_dataset_ae
from continual_learning.experiments.ae_config import (
    DATA_FILE, RESULTS_BASE, get_device, EPOCHS_PER_WINDOW, NUM_WINDOWS,
    AE_ONLINE_MODEL_CONFIGS, AE_OFFLINE_REFERENCE, ER_SCALED_DEFAULTS
)


def run(model_name="base"):
    """Run Scaled ER experiment for a single model size."""
    device = get_device()
    print("[Environment] Device: {}".format(device))

    print("[Data] Loading dataset: {}".format(DATA_FILE))
    dataset = WindowedAEDataset(DATA_FILE, num_windows=NUM_WINDOWS)

    config = AE_ONLINE_MODEL_CONFIGS[model_name]
    model = LinearAutoEncoder(
        input_dim=dataset.window_input_dim,
        encoder_hidden=config['encoder_hidden'],
        decoder_hidden=config['decoder_hidden'],
        latent_dim=config['latent_dim'],
        dropout=config['dropout'],
    ).to(device)

    strategy = ExperienceReplayStrategy(**ER_SCALED_DEFAULTS)
    output_dir = os.path.join(RESULTS_BASE, "er_scaled_{}".format(model_name))

    # Train
    metrics = train_online_ae_cl(
        model=model, dataset=dataset, device=device,
        epochs_per_window=EPOCHS_PER_WINDOW,
        model_name="{}_er_scaled".format(model_name),
        output_dir=output_dir, strategy=strategy,
        num_windows=NUM_WINDOWS,
    )

    # Save per-window metrics
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(output_dir, "{}_er_scaled_metrics.csv".format(model_name)),
              index=False)

    # Full dataset evaluation
    full_eval = evaluate_full_dataset_ae(
        model, dataset, device, model_name="{}_er_scaled".format(model_name)
    )

    # Save evaluation
    eval_data = {
        "model": "{}_er_scaled".format(model_name),
        "training_mode": "online_streaming",
        "strategy": strategy.get_config(),
        "parameters": sum(p.numel() for p in model.parameters()),
        **full_eval,
    }
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(eval_data, f, indent=2)

    # Forgetting gap
    ref = AE_OFFLINE_REFERENCE[model_name]
    print("[Comparison] Offline AE reference PSNR: {:.2f} dB".format(ref['psnr_db']))
    print("[Comparison] Gap to offline: {:.2f} dB".format(
        full_eval['psnr_db'] - ref['psnr_db']))

    return {
        "model": model_name,
        "last_window_psnr": metrics["psnr"][-1],
        "full_dataset_psnr": full_eval["psnr_db"],
        "full_dataset_ssim": full_eval["ssim"],
        "full_dataset_re": full_eval["relative_error_pct"],
        "total_time": sum(metrics["time_per_window"]),
    }


def run_all():
    """Run Scaled ER across all model sizes."""
    results = []
    for model_name in AE_ONLINE_MODEL_CONFIGS:
        result = run(model_name)
        results.append(result)

    print("\n[Summary] Online AE — ER Scaled (buffer={}, weight={})".format(
        ER_SCALED_DEFAULTS['buffer_size'], ER_SCALED_DEFAULTS['replay_weight']))
    print("{:<10} {:>12} {:>10} {:>8} {:>8} {:>8}".format(
        'Model', 'Window PSNR', 'Full PSNR', 'SSIM', 'RE%', 'Time'))
    print("-" * 60)
    for r in results:
        print("{:<10} {:>10.2f} dB {:>8.2f} dB {:>8.4f} {:>7.2f}% {:>7.1f}s".format(
            r['model'], r['last_window_psnr'], r['full_dataset_psnr'],
            r['full_dataset_ssim'], r['full_dataset_re'], r['total_time']))

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Online AE — Scaled Experience Replay")
    parser.add_argument("--model", type=str, default=None,
                        choices=["base", "medium", "large"],
                        help="Run single model (default: all)")
    args = parser.parse_args()

    if args.model:
        run(args.model)
    else:
        run_all()
