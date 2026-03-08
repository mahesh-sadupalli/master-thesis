"""
Run Continual Learning Experiments

Trains all model sizes (base, medium, large) with each continual learning
strategy and evaluates on the full dataset to measure forgetting mitigation.

Usage:
    python run_experiments.py                    # Run all strategies, all models
    python run_experiments.py --strategy er      # Run only Experience Replay
    python run_experiments.py --model base       # Run only base model
    python run_experiments.py --strategy er --model base  # Single combo
"""

import sys
import os
import argparse
import torch
import pandas as pd
import json
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unified_training_utils import (
    SpatioTemporalDataset,
    BaseCompressor,
    MediumCompressor,
    LargeCompressor,
)
from continual_learning.cl_strategies import (
    NaiveStrategy,
    ExperienceReplayStrategy,
    EWCStrategy,
    LwFStrategy,
    DERppStrategy,
    CombinedStrategy,
)
from continual_learning.cl_training import train_online_cl, evaluate_full_dataset


# === Configuration ===

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                         'ML_test_loader_original_data.csv')
RESULTS_BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'results',
                            'continual_learning')

EPOCHS_PER_WINDOW = 100
NUM_WINDOWS = 20

MODELS = {
    'base': BaseCompressor,
    'medium': MediumCompressor,
    'large': LargeCompressor,
}

# Strategy factories with default hyperparameters
STRATEGIES = {
    'naive': lambda: NaiveStrategy(),
    'er': lambda: ExperienceReplayStrategy(
        buffer_size=10000, replay_weight=0.5, replay_batch_size=5000
    ),
    'ewc': lambda: EWCStrategy(
        ewc_lambda=1000.0, fisher_samples=50000
    ),
    'lwf': lambda: LwFStrategy(
        distill_weight=0.5
    ),
    'der_pp': lambda: DERppStrategy(
        buffer_size=10000, replay_weight=0.5, distill_weight=0.5,
        replay_batch_size=5000
    ),
    'combined': lambda: CombinedStrategy(
        buffer_size=10000, replay_weight=0.3, distill_weight=0.3,
        ewc_lambda=500.0, replay_batch_size=5000
    ),
    # Scaled ER variants: buffer scales with model capacity
    'er_scaled': lambda: ExperienceReplayStrategy(
        buffer_size=50000, replay_weight=0.7, replay_batch_size=10000
    ),
    # Aggressive ER: large buffer + high replay weight
    'er_aggressive': lambda: ExperienceReplayStrategy(
        buffer_size=100000, replay_weight=1.0, replay_batch_size=20000
    ),
}


def run_single_experiment(model_name, strategy_name, dataset, device):
    """Run a single model + strategy experiment."""
    print(f"\n[Experiment] model={model_name}, strategy={strategy_name}")

    # Create fresh model and strategy
    model = MODELS[model_name]().to(device)
    strategy = STRATEGIES[strategy_name]()

    output_dir = os.path.join(RESULTS_BASE, f'{model_name}_{strategy_name}')

    # Train
    metrics = train_online_cl(
        model=model,
        dataset=dataset,
        device=device,
        epochs_per_window=EPOCHS_PER_WINDOW,
        model_name=f'{model_name}_{strategy_name}',
        output_dir=output_dir,
        strategy=strategy,
        num_windows=NUM_WINDOWS,
    )

    # Save per-window metrics CSV
    df = pd.DataFrame(metrics)
    csv_path = os.path.join(output_dir, f'{model_name}_{strategy_name}_metrics.csv')
    df.to_csv(csv_path, index=False)

    # Evaluate on full dataset (the forgetting test)
    full_eval = evaluate_full_dataset(
        model, dataset, device,
        model_name=f'{model_name}_{strategy_name}'
    )

    # Save full evaluation
    eval_path = os.path.join(output_dir, 'evaluation_metrics.json')
    eval_data = {
        'model': f'{model_name}_{strategy_name}',
        'architecture': str(model),
        'training_mode': 'online_streaming',
        'strategy': strategy.get_config(),
        'parameters': sum(p.numel() for p in model.parameters()),
        **full_eval
    }
    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, indent=2)

    return {
        'model': model_name,
        'strategy': strategy_name,
        'last_window_psnr': metrics['psnr'][-1],
        'last_window_ssim': metrics['ssim'][-1],
        'last_window_re': metrics['relative_error'][-1],
        'full_dataset_psnr': full_eval['psnr_db'],
        'full_dataset_ssim': full_eval['ssim'],
        'full_dataset_re': full_eval['relative_error_pct'],
        'psnr_drop': metrics['psnr'][-1] - full_eval['psnr_db'],
        'total_time': sum(metrics['time_per_window']),
    }


def main():
    parser = argparse.ArgumentParser(description='Run CL experiments')
    parser.add_argument('--strategy', type=str, default=None,
                        choices=list(STRATEGIES.keys()),
                        help='Run only this strategy (default: all)')
    parser.add_argument('--model', type=str, default=None,
                        choices=list(MODELS.keys()),
                        help='Run only this model (default: all)')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load dataset once
    print("Loading dataset...")
    dataset = SpatioTemporalDataset(DATA_FILE)

    # Determine which experiments to run
    strategies = [args.strategy] if args.strategy else list(STRATEGIES.keys())
    models = [args.model] if args.model else list(MODELS.keys())

    # Run experiments
    all_results = []
    total_start = time.time()

    for strategy_name in strategies:
        for model_name in models:
            result = run_single_experiment(model_name, strategy_name, dataset, device)
            all_results.append(result)

    # Summary table
    os.makedirs(RESULTS_BASE, exist_ok=True)
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(RESULTS_BASE, 'experiment_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print(f"\n[Results Summary]")
    print(f"{'Model':<10} {'Strategy':<14} {'Window PSNR':>12} {'Full PSNR':>10} "
          f"{'Drop':>8} {'Full SSIM':>10} {'Full RE%':>9} {'Time':>8}")
    print('-' * 82)

    for r in all_results:
        print(f"{r['model']:<10} {r['strategy']:<14} "
              f"{r['last_window_psnr']:>10.2f} dB "
              f"{r['full_dataset_psnr']:>8.2f} dB "
              f"{r['psnr_drop']:>6.2f} dB "
              f"{r['full_dataset_ssim']:>10.4f} "
              f"{r['full_dataset_re']:>8.2f}% "
              f"{r['total_time']:>7.1f}s")

    total_time = time.time() - total_start
    print(f"\nTotal experiment time: {total_time:.1f}s")
    print(f"Summary saved: {summary_path}")


if __name__ == '__main__':
    main()
