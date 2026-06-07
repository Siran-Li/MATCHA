#!/usr/bin/env python3
"""Create paper-style summary tables and plots from lm-embedding outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from load_datasets import PAPER_PLOT_DATASETS
from result_io import MODEL_SCORES_DIRNAME, SCORES_LONG_FILENAME


DATASET_DISPLAY = {
    "snli": "SNLI",
    "multi_nli": "MultiNLI",
    "truthfulqa": "TruthfulQA",
    "climate_fever": "Climate-FEVER",
    "coco-caption": "COCO-Captions",
    "newts": "NEWTS",
}

CORRECT_COLOR = "#3B6EA8"
INCORRECT_COLOR = "#C75D59"
GAP_COLOR = "#6A7D4F"


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
    return parser.parse_args()


def main() -> None:
    """Build summary table and optional plots."""
    args = parse_args()
    figures_dir = args.figures_dir or args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    scores = load_scores(args.output_dir, args.datasets)
    if scores.empty:
        raise ValueError(f"No score rows found in {args.output_dir}")

    summary = summarize_scores(scores)
    summary_path = figures_dir / "summary_table.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    if not args.no_barplots:
        barplot_dir = figures_dir / "embedding_barplots_new"
        barplot_dir.mkdir(parents=True, exist_ok=True)
        for dataset in args.datasets:
            dataset_scores = scores[scores["dataset"] == dataset]
            if dataset_scores.empty:
                continue
            display = DATASET_DISPLAY.get(dataset, dataset)
            plot_similarity_bar(dataset_scores, barplot_dir / f"{display}_mean_similarity_barplot.png")


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
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = summarize_scores(scores)
    labels = summary["model_display"].fillna(summary["model"]).astype(str).tolist()
    x = np.arange(len(summary))
    width = 0.25

    fig_width = max(8.0, len(summary) * 0.85)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.bar(x - width, summary["correct_mean"], width, label="Correct", color=CORRECT_COLOR, hatch="//")
    ax.bar(x, summary["incorrect_mean"], width, label="Incorrect", color=INCORRECT_COLOR, hatch="\\\\")
    ax.bar(x + width, summary["gap_mean"], width, label="Gap", color=GAP_COLOR, hatch="..")

    dataset = str(summary["dataset"].iloc[0])
    ax.set_title(DATASET_DISPLAY.get(dataset, dataset))
    ax.set_ylabel("Mean cosine similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
