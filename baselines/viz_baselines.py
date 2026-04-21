"""
viz_baselines.py — Plot threshold-based correct/incorrect separation curves.

For each triplet dataset, plots the percentage of samples where:
  (correct_score - incorrect_score) > threshold
  AND incorrect_score < 0.5 AND correct_score > 0.5
as the threshold varies from 0 to 1.

Only supports triplet datasets (paired correct/incorrect per row).
Pairwise datasets (e.g. climate_fever) are skipped automatically.

Reads the rescaled per-sample CSVs produced by eval_baselines.py.

Usage:
    python viz_baselines.py --data-dir ../data/convert/baselines
    python viz_baselines.py --data-dir ../data/convert/baselines --datasets snli multi_nli
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset / metric configuration (must match eval_baselines.py)
# ---------------------------------------------------------------------------
TRIPLET_DATASETS = [
    'snli', 'multi_nli', 'truthfulqa',
    'coco-caption', 'newts', 'mednli'
]
PAIRWISE_DATASETS = ['climate_fever']

DATASET_DISPLAY = {
    'snli': 'SNLI', 'multi_nli': 'MultiNLI', 'mednli': 'MedNLI',
    'truthfulqa': 'TruthfulQA', 'coco-caption': 'COCO-Caption',
    'newts': 'NEWTS',
}

METRIC_PAIRS = [
    ('r1f1',    'R1-F1',     'green'),
    ('r2f1',    'R2-F1',     'red'),
    ('rLf1',    'RL-F1',     'purple'),
    ('meteor',  'METEOR',    'brown'),
    ('embsim',  'EmbSim',    'orange'),
    ('bert',    'BertScore',  'pink'),
    ('bleurt',  'BLEURT',    'gray'),
    ('simcse',  'SimCSE',    'palegreen'),
    ('matcha',  'MATCHA',    'blue'),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot threshold separation curves")
    parser.add_argument('--data-dir', required=True,
                        help='Path to eval_baselines.py output dir (contains {dataset}/{dataset}.csv)')
    parser.add_argument('--datasets', nargs='*', default=None,
                        help=f'Triplet datasets to plot. Default: all. Choices: {TRIPLET_DATASETS}')
    parser.add_argument('--output-name', default='combined_threshold',
                        help='Output filename prefix (default: combined_threshold)')
    parser.add_argument('--incorrect-threshold', type=float, default=0.5,
                        help='Threshold below which incorrect score must fall (default: 0.5)')
    return parser.parse_args()


def compute_threshold_curve(correct_scores, incorrect_scores, thresholds, incorrect_threshold):
    """Compute percentage of samples satisfying the threshold condition."""
    num_data = len(correct_scores)
    percentages = []
    for t in thresholds:
        condition = (
            (correct_scores - incorrect_scores > t)
            & (incorrect_scores < incorrect_threshold)
            & (correct_scores > incorrect_threshold)
        )
        percentages.append(condition.sum() / num_data * 100)
    return percentages


if __name__ == "__main__":
    args = parse_args()
    datasets = args.datasets if args.datasets else TRIPLET_DATASETS
    incorrect_threshold = args.incorrect_threshold

    # Exclude pairwise datasets (no paired correct/incorrect per row)
    skipped = [d for d in datasets if d in PAIRWISE_DATASETS]
    if skipped:
        print(f"Skipping pairwise datasets (no paired rows): {skipped}")
    datasets = [d for d in datasets if d not in PAIRWISE_DATASETS]

    # Filter to datasets that actually have data
    available = []
    for d in datasets:
        csv_path = os.path.join(args.data_dir, d, f'{d}.csv')
        if os.path.exists(csv_path):
            available.append(d)
        else:
            print(f"Warning: {csv_path} not found, skipping {d}")
    datasets = available

    if not datasets:
        print("No datasets found. Exiting.")
        exit(1)

    # Plot settings
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(4 * n_datasets, 5), sharey=True)
    if n_datasets == 1:
        axes = [axes]
    plt.subplots_adjust(wspace=0.3)

    thresholds = np.linspace(0, 1, 100)

    for i, dataset in enumerate(datasets):
        csv_path = os.path.join(args.data_dir, dataset, f'{dataset}.csv')
        result_df = pd.read_csv(csv_path)
        num_data = len(result_df)
        ax = axes[i]

        for metric_key, label, color in METRIC_PAIRS:
            if metric_key == 'matcha':
                correct_col = 'correct_matcha'
                incorrect_col = 'incorrect_matcha'
            else:
                correct_col = f'correct_{metric_key}'
                incorrect_col = f'incorrect_{metric_key}'

            if correct_col not in result_df.columns:
                print(f"  Warning: column '{correct_col}' not in {dataset}.csv, skipping {label}")
                continue

            df_clean = result_df[[correct_col, incorrect_col]].dropna()
            percentages = compute_threshold_curve(
                df_clean[correct_col], df_clean[incorrect_col],
                thresholds, incorrect_threshold,
            )
            ax.plot(thresholds, percentages, label=label, color=color, linewidth=2)

        ax.set_xlabel('Threshold (correct - incorrect)', fontsize=14)
        ax.set_ylabel('Percentage (%)' if i == 0 else '', fontsize=14)
        ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=16, pad=10)
        ax.grid(True, linestyle='--', alpha=0.7)

        if i == n_datasets - 1:
            ax.legend(framealpha=1, fontsize=12)

    plt.tight_layout()

    output_dir = os.path.join(args.data_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{args.output_name}_incorr{incorrect_threshold}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
