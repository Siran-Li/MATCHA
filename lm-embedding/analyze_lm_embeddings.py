#!/usr/bin/env python3
"""Create paper-style summary tables and plots from lm-embedding outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from load_datasets import PAPER_PLOT_DATASETS
from result_io import MODEL_SCORES_DIRNAME, SCORES_LONG_FILENAME

DEFAULT_THRESHOLD_DATASETS = ("snli", "multi_nli", "truthfulqa")


DATASET_DISPLAY = {
    "snli": "SNLI",
    "multi_nli": "MultiNLI",
    "truthfulqa": "TruthfulQA",
    "climate_fever": "Climate-FEVER",
    "coco-caption": "COCO-Captions",
    "newts": "NEWTS",
}

CORRECT_COLOR = "#1f77b4"
INCORRECT_COLOR = "#ff7f0e"
GAP_COLOR = "#2ca02c"
CORRECT_HATCH = "///"
INCORRECT_HATCH = "\\\\\\"
GAP_HATCH = "xxxx"
GRID_STYLE = {"linestyle": "--", "linewidth": 0.7, "alpha": 0.5, "color": "gray"}
PAPER_THRESHOLDS = np.linspace(0, 2, 200)
THRESHOLD_FILENAME_SUFFIX = "threshold_curves_norm=False_cutoff=None.png"
THRESHOLD_COLOR_LABELS = [
    "Word2Vec",
    "GloVe",
    "SNPMI",
    "BERT (base)",
    "MiniLM (S-BERT)",
    "MPNet",
    "DistilBERT-NLI",
    "T5-Large (Sentence-T5)",
    "E5-Large-v2",
    "BGE-Large",
    "GTE-Large",
    "Mistral-7B",
    "LLaMA-2-13B",
    "LLaMA-3.1-8B",
    "E5-Mistral-7B",
    "SpeedEmbed-7B-Instruct",
    "SFR-Mistral",
    "Linq-Embed-Mistral",
    "E5-Large (Multilingual)",
    "Jasper",
    "Stella",
    "Bilingual-Embedding-Large",
    "Jina-Embeddings-V3",
    "MATCHA",
]
PREDEFINED_THRESHOLD_COLORS = {
    "DistilBERT-NLI": "black",
    "MPNet": "#daa520",
    "MiniLM (S-BERT)": "#984ea3",
    "Jina-Embeddings-V3": "#ff7f00",
    "SFR-Mistral": "#a65628",
    "T5-Large (Sentence-T5)": "#7B002C",
    "MATCHA": "blue",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze generated lm-embedding score tables."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "test_all_models_all_datasets",
        help="Directory containing per-dataset lm-embedding outputs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(PAPER_PLOT_DATASETS),
        help="Datasets to include. Default: paper plot datasets only.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory for analysis artifacts. Default: OUTPUT_DIR/figures.",
    )
    parser.add_argument(
        "--no-barplots",
        action="store_true",
        help="Skip per-dataset mean similarity barplots.",
    )
    parser.add_argument(
        "--threshold-datasets",
        nargs="*",
        default=list(DEFAULT_THRESHOLD_DATASETS),
        help="Datasets to include in threshold curves.",
    )
    parser.add_argument(
        "--no-threshold-curves",
        action="store_true",
        help="Skip paper-style threshold curves.",
    )
    return parser.parse_args()


def main() -> None:
    """Build summary table and optional plots."""
    args = parse_args()
    figures_dir = args.figures_dir or args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    requested = unique_datasets(
        list(args.datasets)
        + ([] if args.no_threshold_curves else list(args.threshold_datasets))
    )
    scores = load_scores(args.output_dir, requested)
    if scores.empty:
        raise ValueError(f"No score rows found in {args.output_dir}")

    analysis_scores = filter_available(scores, args.datasets)
    if analysis_scores.empty:
        raise ValueError(f"No requested dataset rows found in {args.output_dir}")

    summary = summarize_scores(analysis_scores)
    summary_path = figures_dir / "summary_table.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    if not args.no_barplots:
        barplot_dir = figures_dir / "embedding_barplots"
        barplot_dir.mkdir(parents=True, exist_ok=True)
        for dataset in args.datasets:
            dataset_scores = analysis_scores[analysis_scores["dataset"] == dataset]
            if dataset_scores.empty:
                continue
            display = DATASET_DISPLAY.get(dataset, dataset)
            plot_similarity_bar(dataset_scores, barplot_dir / f"{display}_mean_similarity_barplot.png")

    if not args.no_threshold_curves:
        threshold_scores = filter_available(scores, args.threshold_datasets)
        if not threshold_scores.empty:
            plot_threshold_curves(threshold_scores, figures_dir / "threshold_curves")


def load_scores(output_dir: Path, datasets: list[str]) -> pd.DataFrame:
    """Load scores_long.csv or model_scores/*.csv for the selected datasets."""
    frames = []
    for dataset in datasets:
        dataset_dir = output_dir / dataset
        scores_long = dataset_dir / SCORES_LONG_FILENAME
        if scores_long.exists():
            frames.append(pd.read_csv(scores_long))
            continue

        model_scores_dir = dataset_dir / MODEL_SCORES_DIRNAME
        if not model_scores_dir.exists():
            continue
        model_frames = [
            pd.read_csv(path)
            for path in sorted(model_scores_dir.glob("*.csv"))
        ]
        if model_frames:
            frames.append(pd.concat(model_frames, ignore_index=True))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def filter_available(scores: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    """Keep requested datasets that are present in the score table."""
    available = []
    seen_display = set()
    for dataset in requested:
        subset = scores[scores["dataset"] == dataset]
        display = DATASET_DISPLAY.get(dataset, dataset)
        if subset.empty or display in seen_display:
            continue
        available.append(subset)
        seen_display.add(display)
    return pd.concat(available, ignore_index=True) if available else pd.DataFrame()


def unique_datasets(datasets: list[str]) -> list[str]:
    """Keep dataset order while dropping repeats."""
    return list(dict.fromkeys(datasets))


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize score rows by dataset/model."""
    rows = []
    grouped = scores.groupby(
        ["dataset", "model", "model_display", "embedding_details"],
        dropna=False,
    )
    for keys, group in grouped:
        correct_mean = group["correct_sim"].mean()
        incorrect_mean = group["incorrect_sim"].mean()
        gap_mean = group["gap"].mean()
        if pd.isna(gap_mean) and pd.notna(correct_mean) and pd.notna(incorrect_mean):
            gap_mean = correct_mean - incorrect_mean
        rows.append({
            "dataset": keys[0],
            "model": keys[1],
            "model_display": keys[2],
            "embedding_details": keys[3],
            "n_rows": int(group["row_id"].count()),
            "correct_mean": correct_mean,
            "incorrect_mean": incorrect_mean,
            "gap_mean": gap_mean,
        })

    summary = pd.DataFrame(rows)
    summary["dataset_order"] = summary["dataset"].map({name: i for i, name in enumerate(PAPER_PLOT_DATASETS)})
    summary = summary.sort_values(["dataset_order", "model_display"]).drop(columns=["dataset_order"])
    return summary.reset_index(drop=True)


def plot_similarity_bar(scores: pd.DataFrame, output_path: Path) -> None:
    """Plot paper-style mean correct/incorrect/gap bars for one dataset."""
    _, plt = load_matplotlib()

    summary = summarize_scores(scores)
    summary["_sort_key"] = (
        summary["model_display"].fillna(summary["model"]).astype(str).map(model_label_sort_key)
    )
    summary = summary.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)
    plot_data = summary.copy()
    plot_data["correct_mean"] = plot_data["correct_mean"] * 100
    plot_data["incorrect_mean"] = plot_data["incorrect_mean"] * 100
    plot_data["gap_mean"] = plot_data["gap_mean"] * 100

    labels = plot_data["model_display"].fillna(plot_data["model"]).astype(str).tolist()
    x = np.arange(len(plot_data))
    width = 0.2

    fig_width = max(20.0, len(plot_data) * 0.85)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    ax.bar(
        x - width,
        plot_data["correct_mean"],
        width,
        label="Correct",
        color=CORRECT_COLOR,
        edgecolor="black",
        hatch=CORRECT_HATCH,
        linewidth=0.8,
    )
    ax.bar(
        x,
        plot_data["incorrect_mean"],
        width,
        label="Incorrect",
        color=INCORRECT_COLOR,
        edgecolor="black",
        hatch=INCORRECT_HATCH,
        linewidth=0.8,
    )
    ax.bar(
        x + width,
        plot_data["gap_mean"],
        width,
        label="Gap",
        color=GAP_COLOR,
        edgecolor="black",
        hatch=GAP_HATCH,
        linewidth=0.8,
    )

    dataset = str(summary["dataset"].iloc[0])
    dataset_label = DATASET_DISPLAY.get(dataset, dataset)
    ax.set_title(f"Mean Similarity - Correct vs Incorrect vs Gap ({dataset_label})")
    ax.set_ylabel("Mean Similarity (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, **GRID_STYLE)
    ax.legend(framealpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_threshold_curves(scores: pd.DataFrame, output_dir: Path) -> None:
    """Plot the paper threshold curves with fixed raw-score settings."""
    _, plt = load_matplotlib()

    datasets = list(dict.fromkeys(scores["dataset"].tolist()))
    limit = len(datasets)
    fig, axes = plt.subplots(1, limit, figsize=(20, 6), sharey=True)
    if limit == 1:
        axes = [axes]
    plt.subplots_adjust(wspace=limit * 0.1)

    labels_for_colors = score_labels(scores)
    color_map = threshold_color_map(labels_for_colors)
    legend_handles = {}

    for ax, dataset in zip(axes, datasets):
        data = scores[scores["dataset"] == dataset]
        for (model, model_display), group in data.groupby(
            ["model", "model_display"], dropna=False
        ):
            gaps = group["gap"].astype(float).to_numpy()
            if len(gaps) == 0:
                continue
            percentages = [
                np.sum(gaps > threshold) / len(group) * 100
                for threshold in PAPER_THRESHOLDS
            ]
            label = model_display if pd.notna(model_display) else model
            line = ax.plot(
                PAPER_THRESHOLDS,
                percentages,
                label=label,
                linewidth=2,
                color=color_map.get(label, "gray"),
            )[0]
            legend_handles.setdefault(label, line)

        ax.set_title(DATASET_DISPLAY.get(dataset, dataset))
        ax.set_xlabel("Threshold (Correct - Incorrect)")
        if ax == axes[0]:
            ax.set_ylabel("Percentage (%)")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 100)

    if legend_handles:
        ordered = sorted(legend_handles.items(), key=lambda item: model_label_sort_key(item[0]))
        labels, handles = zip(*ordered)
        axes[-1].legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / threshold_filename(datasets)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {output_path}")


def load_matplotlib():
    """Load matplotlib with a writable config directory."""
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def threshold_filename(datasets: list[str]) -> str:
    """Return the threshold-curve filename for the plotted dataset set."""
    dataset_part = "_".join(datasets)
    return f"{dataset_part}_{THRESHOLD_FILENAME_SUFFIX}"


def score_labels(scores: pd.DataFrame) -> list[str]:
    """Return model labels present in the score table."""
    labels = scores["model_display"].fillna(scores["model"]).astype(str)
    return labels.tolist()


def threshold_color_map(labels: list[str]) -> dict[str, str]:
    """Build the seeded color map used by threshold curves."""
    matplotlib, _ = load_matplotlib()
    import seaborn as sns

    np.random.seed(42)
    aliases = sorted(set(THRESHOLD_COLOR_LABELS) | set(labels))
    color_pool = list(sns.color_palette("hsv", len(aliases) + 10))
    np.random.shuffle(color_pool)

    color_map = {}
    used_colors = set()
    for alias in aliases:
        if alias in PREDEFINED_THRESHOLD_COLORS:
            color_map[alias] = PREDEFINED_THRESHOLD_COLORS[alias]
            used_colors.add(PREDEFINED_THRESHOLD_COLORS[alias].lower())
            continue

        while color_pool:
            candidate = matplotlib.colors.to_hex(color_pool.pop(0))
            if candidate.lower() not in used_colors:
                color_map[alias] = candidate
                used_colors.add(candidate.lower())
                break
        else:
            raise RuntimeError("Ran out of distinct colors")
    return color_map


def model_label_sort_key(label: str) -> tuple[int, str]:
    """Sort labels alphabetically, with MATCHA pinned after all baselines."""
    return (1 if label == "MATCHA" else 0, label.casefold())


if __name__ == "__main__":
    main()
