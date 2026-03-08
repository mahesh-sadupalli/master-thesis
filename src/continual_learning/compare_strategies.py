"""
Compare Continual Learning Strategies

Generates comparison plots and tables from experiment results.
Reads results from results/continual_learning/ directory.

Usage:
    python compare_strategies.py
"""

import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

RESULTS_BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'results',
                            'continual_learning')


def load_all_results():
    """Load evaluation and training metrics from all experiment directories."""
    results = []

    if not os.path.exists(RESULTS_BASE):
        print(f"No results found at {RESULTS_BASE}")
        return pd.DataFrame()

    for dirname in sorted(os.listdir(RESULTS_BASE)):
        dirpath = os.path.join(RESULTS_BASE, dirname)
        if not os.path.isdir(dirpath):
            continue

        eval_path = os.path.join(dirpath, 'evaluation_metrics.json')
        summary_path = os.path.join(dirpath, 'online_training_summary.json')

        if not os.path.exists(eval_path):
            continue

        with open(eval_path, 'r') as f:
            eval_data = json.load(f)

        training_time = None
        last_window_psnr = None
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                training_time = summary.get('total_training_time')
                last_window_psnr = summary.get('final_psnr')

        # Parse model and strategy from directory name
        parts = dirname.split('_')
        model_name = parts[0]  # base, medium, large
        strategy_name = '_'.join(parts[1:])

        results.append({
            'model': model_name,
            'strategy': strategy_name,
            'parameters': eval_data.get('parameters', 0),
            'full_psnr': eval_data['psnr_db'],
            'full_ssim': eval_data['ssim'],
            'full_re': eval_data['relative_error_pct'],
            'last_window_psnr': last_window_psnr,
            'psnr_drop': (last_window_psnr - eval_data['psnr_db']) if last_window_psnr else None,
            'training_time': training_time,
            'dirname': dirname,
        })

    return pd.DataFrame(results)


def plot_strategy_comparison(df):
    """Create bar charts comparing strategies across model sizes."""
    if df.empty:
        print("No data to plot")
        return

    output_dir = os.path.join(RESULTS_BASE, 'comparison_plots')
    os.makedirs(output_dir, exist_ok=True)

    strategies = df['strategy'].unique()
    models = ['base', 'medium', 'large']
    models = [m for m in models if m in df['model'].values]

    x = np.arange(len(models))
    width = 0.8 / len(strategies)

    # --- PSNR Comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Full dataset PSNR
    for i, strat in enumerate(strategies):
        subset = df[df['strategy'] == strat].set_index('model')
        values = [subset.loc[m, 'full_psnr'] if m in subset.index else 0 for m in models]
        axes[0].bar(x + i * width, values, width, label=strat)

    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('Full Dataset PSNR (Higher = Better)', fontweight='bold')
    axes[0].set_xticks(x + width * (len(strategies) - 1) / 2)
    axes[0].set_xticklabels(models)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Full dataset SSIM
    for i, strat in enumerate(strategies):
        subset = df[df['strategy'] == strat].set_index('model')
        values = [subset.loc[m, 'full_ssim'] if m in subset.index else 0 for m in models]
        axes[1].bar(x + i * width, values, width, label=strat)

    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('Full Dataset SSIM (Higher = Better)', fontweight='bold')
    axes[1].set_xticks(x + width * (len(strategies) - 1) / 2)
    axes[1].set_xticklabels(models)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')

    # PSNR drop (forgetting severity)
    for i, strat in enumerate(strategies):
        subset = df[df['strategy'] == strat].set_index('model')
        values = [subset.loc[m, 'psnr_drop'] if m in subset.index and subset.loc[m, 'psnr_drop'] is not None else 0 for m in models]
        axes[2].bar(x + i * width, values, width, label=strat)

    axes[2].set_xlabel('Model')
    axes[2].set_ylabel('PSNR Drop (dB)')
    axes[2].set_title('Forgetting Severity (Lower = Better)', fontweight='bold')
    axes[2].set_xticks(x + width * (len(strategies) - 1) / 2)
    axes[2].set_xticklabels(models)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Continual Learning Strategy Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/strategy_comparison.png")
    plt.close()


def plot_training_curves(df):
    """Plot per-window training curves for all strategies (per model)."""
    output_dir = os.path.join(RESULTS_BASE, 'comparison_plots')
    os.makedirs(output_dir, exist_ok=True)

    models = ['base', 'medium', 'large']
    models = [m for m in models if m in df['model'].values]

    for model_name in models:
        model_df = df[df['model'] == model_name]
        strategies = model_df['strategy'].unique()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for strat in strategies:
            dirname = model_df[model_df['strategy'] == strat]['dirname'].values[0]
            metrics_files = [
                os.path.join(RESULTS_BASE, dirname,
                             f'{model_name}_{strat}_metrics.csv'),
            ]

            for mf in metrics_files:
                if os.path.exists(mf):
                    mdata = pd.read_csv(mf)
                    axes[0].plot(mdata['window'], mdata['psnr'], '-o',
                                label=strat, markersize=3, linewidth=1.5)
                    axes[1].plot(mdata['window'], mdata['relative_error'], '-o',
                                label=strat, markersize=3, linewidth=1.5)
                    break

        axes[0].set_xlabel('Window')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title('Per-Window PSNR During Training')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Window')
        axes[1].set_ylabel('Relative Error (%)')
        axes[1].set_title('Per-Window Relative Error During Training')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'{model_name.capitalize()} Model - Training Curves by Strategy',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_training_curves.png'),
                    dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/{model_name}_training_curves.png")
        plt.close()


def main():
    df = load_all_results()

    if df.empty:
        print("No experiment results found. Run run_experiments.py first.")
        return

    # Print summary table
    print(f"\n{'='*90}")
    print("FULL DATASET EVALUATION (Forgetting Test)")
    print(f"{'='*90}")
    print(f"\n{'Model':<10} {'Strategy':<12} {'Params':>8} {'Full PSNR':>10} "
          f"{'Full SSIM':>10} {'Full RE%':>9} {'Drop':>8} {'Time':>8}")
    print('-' * 90)

    for _, r in df.iterrows():
        drop_str = f"{r['psnr_drop']:.2f} dB" if r['psnr_drop'] is not None else "N/A"
        time_str = f"{r['training_time']:.1f}s" if r['training_time'] is not None else "N/A"
        print(f"{r['model']:<10} {r['strategy']:<12} {r['parameters']:>8,} "
              f"{r['full_psnr']:>8.2f} dB "
              f"{r['full_ssim']:>10.4f} "
              f"{r['full_re']:>8.2f}% "
              f"{drop_str:>8} "
              f"{time_str:>8}")

    # Generate plots
    plot_strategy_comparison(df)
    plot_training_curves(df)

    print(f"\nAll comparison plots saved to: {RESULTS_BASE}/comparison_plots/")


if __name__ == '__main__':
    main()
