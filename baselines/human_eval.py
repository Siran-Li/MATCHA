"""
human_eval.py — Map baseline + MATCHA metrics to human scores and compute
correlation, R@k, and distribution analyses.

Reads the rescaled per-sample CSVs from eval_baselines.py, flattens triplet
rows into per-sentence rows, merges with human_scores.csv, and runs:
  - Correlation analysis with heatmaps
  - R@1 / MAE / DCG analysis
  - Score distribution plots (correct vs incorrect)

Usage:
    python baselines/human_eval.py \
        --baselines-dir /mnt/lustre/work/eickhoff/esx400/metrics/MATCHA_best/test/baselines \
        --human-scores data/human_scores/human_scores.csv \
        --analysis correlation_metrics
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Datasets that have human annotations (triplet only)
HUMAN_DATASETS = {
    'snli': 'SNLI',
    'multi_nli': 'MultiNLI',
    'truthfulqa': 'TruthfulQA',
}

# Metric columns in the baselines CSV (after flattening)
METRICS = ['r1f1', 'r2f1', 'rLf1', 'meteor', 'embsim', 'bert', 'bleurt', 'simcse', 'matcha']

# Metrics that need rescaling from [-1,1] to [0,1]
RESCALE_METRICS = ['embsim', 'bleurt', 'simcse', 'matcha']

LABEL_MAP = {
    'r1f1': 'R1-F1', 'r2f1': 'R2-F1', 'rLf1': 'RL-F1',
    'meteor': 'METEOR', 'embsim': 'EmbSim', 'bert': 'BERTScore',
    'bleurt': 'BLEURT', 'simcse': 'SimCSE', 'matcha': 'MATCHA',
    'HumanScore': 'Human',
}


def parse_args():
    parser = argparse.ArgumentParser(description="Human evaluation analysis")
    parser.add_argument('--baselines-dir', required=True,
                        help='Path to eval_baselines.py output (contains {dataset}/{dataset}.csv)')
    parser.add_argument('--human-scores', required=True,
                        help='Path to human_scores.csv')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: {baselines-dir}/human_eval)')
    parser.add_argument('--analysis', default='all',
                        choices=['correlation_metrics', 'r@k', 'distribution', 'all'],
                        help='Analysis type to run')
    parser.add_argument('--threshold', type=float, default=0.00,
                        help='Threshold for R@1 (default: 0.00)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Flatten triplet rows to per-sentence rows
# ---------------------------------------------------------------------------

def flatten_triplet_csv(csv_path, dataset_name):
    """Read a baselines triplet CSV and flatten to per-sentence rows.

    Each triplet row becomes two rows: one for correct, one for incorrect,
    with columns: dataset, reference, sentence, label, and metric scores.
    """
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        base = {'dataset': dataset_name, 'reference': row['reference']}

        # Correct row
        correct = base.copy()
        correct['sentence'] = row['correct']
        correct['label'] = 1
        correct['matcha'] = row['correct_matcha']
        for m in METRICS:
            if m == 'matcha':
                continue
            correct[m] = row[f'correct_{m}']
        rows.append(correct)

        # Incorrect row
        incorrect = base.copy()
        incorrect['sentence'] = row['incorrect']
        incorrect['label'] = 0
        incorrect['matcha'] = row['incorrect_matcha']
        for m in METRICS:
            if m == 'matcha':
                continue
            incorrect[m] = row[f'incorrect_{m}']
        rows.append(incorrect)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def concordance_correlation_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    covar = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * covar) / (var_true + var_pred + (mean_true - mean_pred) ** 2)


def corr_metric(df, save_path, metrics_list, label=''):
    """Compute and save correlation matrices + heatmaps."""
    all_cols = metrics_list + ['HumanScore']

    corr_data = {col: [] for col in ['metric'] + all_cols}

    for test_metric in all_cols:
        corr_data['metric'].append(test_metric)
        for other_metric in all_cols:
            corr = concordance_correlation_coefficient(df[other_metric], df[test_metric])
            corr_data[other_metric].append(corr * 100)

        corr_df = pd.DataFrame(corr_data)
        corr_df.to_csv(os.path.join(save_path, f'Metric_CCC_Correlation.csv'), index=False)

        # Heatmap
        matrix = corr_df.set_index('metric').T
        matrix.index = [LABEL_MAP.get(m, m) for m in matrix.index]
        matrix.columns = [LABEL_MAP.get(m, m) for m in matrix.columns]

        plt.figure(figsize=(8.2, 8))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap='coolwarm',
                    center=0, vmin=-100, vmax=100,
                    linewidths=0.5, linecolor='black')
        plt.title(f'{label}' if label else '', pad=20)
        plt.xticks(rotation=90, fontsize=15)
        plt.yticks(rotation=0, fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'CCC_Heatmap_Human.png'), dpi=300)
        plt.close()


# ---------------------------------------------------------------------------
# R@1 / MAE / DCG
# ---------------------------------------------------------------------------

def calculate_metrics(df, metrics, threshold=0.05):
    dataset_order = list(HUMAN_DATASETS.values()) + ['Overall']
    results = []

    for dataset_name in dataset_order[:-1]:
        if dataset_name not in df['dataset'].unique():
            continue
        group = df[df['dataset'] == dataset_name]
        _compute_rank_stats(group, metrics, dataset_name, threshold, results)

    _compute_rank_stats(df, metrics, 'Overall', threshold, results)

    count_df = pd.DataFrame(results)
    count_df['R@1'] = count_df['R@1'] * 100
    count_df['MAE'] = count_df['MAE'] * 100
    count_df['DCG'] = count_df['DCG'] * 100
    return count_df


def _compute_rank_stats(group, metrics, dataset_name, threshold, results):
    num_samples = len(group)
    human_scores = group['HumanScore'].values
    stats = {m: {'r1': 0, 'mae': 0, 'dcg': 0} for m in metrics}

    for i in range(num_samples):
        human = human_scores[i]
        diffs = {m: abs(group[m].iloc[i] - human) for m in metrics}
        ranked = rankdata(list(diffs.values()))

        for j, m in enumerate(metrics):
            if diffs[m] <= threshold:
                stats[m]['r1'] += 1
                stats[m]['dcg'] += 1 / np.log2(2)
            else:
                if ranked[j] == 1:
                    stats[m]['r1'] += 1
                stats[m]['dcg'] += 1 / np.log2(ranked[j] + 1)
            stats[m]['mae'] += diffs[m]

    for m in metrics:
        results.append({
            'dataset': dataset_name, 'metric': m,
            'R@1': stats[m]['r1'] / num_samples,
            'MAE': stats[m]['mae'] / num_samples,
            'DCG': stats[m]['dcg'] / num_samples,
        })


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------

def plot_distributions(df, metrics, save_path):
    metrics_parts = [
        ['r1f1', 'r2f1', 'rLf1', 'meteor', 'embsim'],
        ['bert', 'bleurt', 'simcse', 'matcha', 'HumanScore'],
    ]

    for part_idx, part_metrics in enumerate(metrics_parts):
        available = [m for m in part_metrics if m in df.columns]
        if not available:
            continue

        fig, axes = plt.subplots(len(available), 1, figsize=(10, 3 * len(available)), sharex=False)
        if len(available) == 1:
            axes = [axes]

        global_min, global_max = -2, 6
        x_pad = (global_max - global_min) * 0.05
        grid_style = dict(linestyle='--', linewidth=0.7, alpha=0.5, color='gray')

        for i, metric in enumerate(available):
            ax = axes[i]
            sns.kdeplot(data=df[df['label'] == 1], x=metric,
                        color='#1f77b4', label='Correct', ax=ax,
                        fill=True, alpha=0.3, hatch='///',
                        edgecolor='#1f77b4', linewidth=1.5)
            sns.kdeplot(data=df[df['label'] == 0], x=metric,
                        color='#ff7f0e', label='Incorrect', ax=ax,
                        fill=True, alpha=0.3, hatch='\\\\\\',
                        edgecolor='#ff7f0e', linewidth=1.5)
            ax.set_xlim(global_min - x_pad, global_max + x_pad)
            ax.set_title(f'Distribution of {LABEL_MAP.get(metric, metric)} scores')
            ax.legend(framealpha=0.7, fontsize=12)
            ax.grid(True, **grid_style)
            ax.set_xlabel('' if i != len(available) - 1 else 'Score')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'distribution_part{part_idx}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.baselines_dir, 'human_eval')

    # Load and flatten baselines data for human-annotated datasets
    all_dfs = []
    for dataset_key, dataset_display in HUMAN_DATASETS.items():
        csv_path = os.path.join(args.baselines_dir, dataset_key, f'{dataset_key}.csv')
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping {dataset_key}")
            continue
        flat_df = flatten_triplet_csv(csv_path, dataset_display)
        print(f"[{dataset_key}] Flattened: {len(flat_df)} rows")
        all_dfs.append(flat_df)

    if not all_dfs:
        print("No data found. Exiting.")
        exit(1)

    metrics_df = pd.concat(all_dfs, axis=0).reset_index(drop=True)
    # Drop duplicate metric rows (same sample may appear in baselines CSV multiple times)
    merge_keys = ['dataset', 'reference', 'sentence', 'label']
    metrics_df = metrics_df.drop_duplicates(subset=merge_keys, keep='first')
    print(f"Metrics (deduped): {len(metrics_df)} rows")

    # Load human scores (keep all annotations including duplicates)
    human_df = pd.read_csv(args.human_scores)
    print(f"Human scores: {len(human_df)} rows")

    # Merge — many-to-one (many human rows per deduped metric row)
    df = human_df.merge(metrics_df, on=merge_keys, how='inner')
    print(f"Merged: {len(df)} rows")

    if len(df) == 0:
        print("No matches after merge. Check column values.")
        exit(1)

    # Rescale scores to [0,1]
    df['HumanScore'] = df['HumanScore'] / 5
    df['bleurt'] = df['bleurt'].clip(lower=0, upper=1)

    # df.drop_duplicates(keep='first')
    print(f"After dedup: {len(df)} rows")
    print(f"Datasets: {df['dataset'].value_counts().to_dict()}")

    analyses = ['correlation_metrics', 'r@k', 'distribution'] if args.analysis == 'all' else [args.analysis]

    for analysis in analyses:
        save_path = os.path.join(output_dir, analysis)
        os.makedirs(save_path, exist_ok=True)

        if analysis == 'correlation_metrics':
            # Overall correlation
            corr_metric(df, save_path, METRICS)
            print(f"Saved correlation analysis to {save_path}")

        elif analysis == 'r@k':
            results_df = calculate_metrics(df, METRICS, threshold=args.threshold)

            output = []
            for metric in METRICS:
                metric_data = {'Metric': metric}
                for ds in list(HUMAN_DATASETS.values()) + ['Overall']:
                    subset = results_df[(results_df['metric'] == metric) & (results_df['dataset'] == ds)]
                    if not subset.empty:
                        metric_data.update({
                            f'{ds} R@1': f"{subset['R@1'].values[0]:.2f}",
                            f'{ds} MAE': f"{subset['MAE'].values[0]:.2f}",
                            f'{ds} AvgDCG': f"{subset['DCG'].values[0]:.2f}",
                        })
                output.append(metric_data)

            final_output = pd.DataFrame(output).set_index('Metric')
            final_output.to_csv(os.path.join(save_path, f'R@k_Metrics_th{args.threshold}.csv'))
            print(f"Saved R@k analysis to {save_path}")
            print(final_output)

        elif analysis == 'distribution':
            plot_distributions(df, METRICS, save_path)
            print(f"Saved distribution plots to {save_path}")

    # Save the merged dataframe
    df.to_csv(os.path.join(output_dir, 'merged_human_metrics.csv'), index=False)
    print(f"\nSaved merged data to {output_dir}/merged_human_metrics.csv")
