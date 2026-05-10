"""Postprocess LM embedding scores into semantic-separation figures"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from result_io import MODEL_SCORES_DIRNAME

DEFAULT_FIG3 = ["snli", "multi_nli", "truthfulqa"]
DEFAULT_FIG5 = ["climate_fever", "coco-caption", "newts"]
DEFAULT_FIG6 = ["snli", "multi_nli", "truthfulqa"]

DATASET_DISPLAY = {
    "snli": "SNLI",
    "multi_nli": "MultiNLI",
    "truthfulqa": "TruthfulQA",
    "climate_fever": "Climate-Fewer",
    "coco-caption": "COCO-Captions",
    "newts": "NEWTS",
}

CORRECT_COLOR = "#1f77b4"
INCORRECT_COLOR = "#ff7f0e"
CORRECT_HATCH = "///"
INCORRECT_HATCH = "\\\\\\"
GRID_STYLE = {"linestyle": "--", "linewidth": 0.7, "alpha": 0.5, "color": "gray"}
PAPER_THRESHOLDS = np.linspace(0, 2, 200)
THRESHOLD_FILENAME = "threshold_curves_norm=False_cutoff=None.png"
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
    """Parse CLI flags for plotting saved score tables"""
    parser = argparse.ArgumentParser(
        description="Plot LM embedding semantic-separation results"
    )
    parser.add_argument(
        "--scores-dir", type=Path, default=Path("outputs/lm_embeddings")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fig3-datasets", nargs="*", default=DEFAULT_FIG3)
    parser.add_argument("--fig5-datasets", nargs="*", default=DEFAULT_FIG5)
    parser.add_argument("--fig6-datasets", nargs="*", default=DEFAULT_FIG6)
    return parser.parse_args()


def main() -> None:
    """Load scores, summarize them, and write the configured figures"""
    args = parse_args()
    output_dir = args.output_dir or (args.scores_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = load_all_scores(args.scores_dir)
    if scores.empty:
        raise ValueError(f"No scores found under {args.scores_dir}")

    summary = summarize(scores)
    summary_path = output_dir / "summary_table.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary table: {summary_path}")

    bar_dir = output_dir / "embedding_barplots_new"
    for dataset in unique_datasets(args.fig3_datasets + args.fig5_datasets):
        dataset_scores = scores[scores["dataset"] == dataset]
        if not dataset_scores.empty:
            plot_similarity_bar(dataset_scores, bar_dir, dataset)

    fig6_scores = filter_available(scores, args.fig6_datasets)
    if not fig6_scores.empty:
        plot_threshold_curves(fig6_scores, output_dir / "threshold_curves")


def load_all_scores(scores_dir: Path) -> pd.DataFrame:
    """Load score CSVs from either combined or per-model output layout"""
    frames = []
    for long_csv in sorted(scores_dir.glob("*/scores_long.csv")):
        frames.append(pd.read_csv(long_csv))

    if frames:
        # Prefer long files when they exist, since they are the scorer's plot-ready output
        return pd.concat(frames, ignore_index=True)

    for model_scores_dir in sorted(scores_dir.glob(f"*/{MODEL_SCORES_DIRNAME}")):
        dataset = model_scores_dir.parent.name
        model_scores = load_per_model_score_files(model_scores_dir, dataset)
        if not model_scores.empty:
            frames.append(model_scores)

    if frames:
        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame()


def load_per_model_score_files(model_scores_dir: Path, dataset: str) -> pd.DataFrame:
    """Load older per-model score CSVs for one dataset"""
    frames = []
    for path in sorted(model_scores_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if "dataset" not in df.columns:
            df["dataset"] = dataset
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """Compute per-dataset/model means used by the bar plots"""
    grouped = scores.groupby(["dataset", "model", "model_display"], dropna=False)
    summary = grouped.agg(
        correct_mean=("correct_sim", "mean"),
        incorrect_mean=("incorrect_sim", "mean"),
        gap_mean=("gap", "mean"),
        n_rows=("row_id", "count"),
    ).reset_index()
    summary["gap_from_means"] = summary["correct_mean"] - summary["incorrect_mean"]
    return summary


def score_label_series(scores: pd.DataFrame) -> pd.Series:
    """Return the display label used for sorting and plotting."""
    return scores["model_display"].fillna(scores["model"]).astype(str)


def model_label_sort_key(label: str) -> tuple[int, str]:
    """Sort labels alphabetically, with MATCHA pinned after all baselines."""
    return (1 if label == "MATCHA" else 0, label.casefold())


def filter_available(scores: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    """Keep requested datasets that are present in the score table"""
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
    """Keep dataset order while dropping repeats"""
    return list(dict.fromkeys(datasets))


def plot_similarity_bar(scores: pd.DataFrame, output_dir: Path, dataset: str) -> None:
    """Plot correct and incorrect mean similarities for one dataset"""
    summary = summarize(scores)
    data = summary[summary["dataset"] == dataset].copy()
    data["_sort_key"] = data["model_display"].map(model_label_sort_key)
    data = data.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)
    data["correct_mean"] = data["correct_mean"] * 100
    data["incorrect_mean"] = data["incorrect_mean"] * 100

    x = np.arange(len(data))
    width = 0.2
    fig, ax = plt.subplots(figsize=(20, 8))

    ax.bar(
        x - width,
        data["correct_mean"],
        width,
        label="Correct",
        facecolor="white",
        edgecolor=CORRECT_COLOR,
        hatch=CORRECT_HATCH,
        linewidth=1.5,
    )
    ax.bar(
        x,
        data["incorrect_mean"],
        width,
        label="Incorrect",
        facecolor="white",
        edgecolor=INCORRECT_COLOR,
        hatch=INCORRECT_HATCH,
        linewidth=1.5,
    )

    labels = data["model_display"].fillna(data["model"]).tolist()
    dataset_label = DATASET_DISPLAY.get(dataset, dataset)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Similarity (%)")
    ax.set_title(f"Mean Similarity - Correct vs Incorrect({dataset_label})")
    ax.legend(framealpha=0.7)
    ax.grid(True, **GRID_STYLE)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_label}_mean_similarity_barplot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved bar figure: {output_path}")


def plot_threshold_curves(scores: pd.DataFrame, output_dir: Path) -> None:
    """Plot the paper threshold curves with fixed raw-score settings"""
    datasets = list(dict.fromkeys(scores["dataset"].tolist()))
    limit = len(datasets)
    fig, axes = plt.subplots(1, limit, figsize=(20, 6), sharey=True)
    if limit == 1:
        axes = [axes]
    plt.subplots_adjust(wspace=limit * 0.1)

    labels_for_colors = score_labels(scores)
    color_map = threshold_color_map(labels_for_colors)

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
            ax.plot(
                PAPER_THRESHOLDS,
                percentages,
                label=label,
                linewidth=2,
                color=color_map.get(label, "gray"),
            )

        ax.set_title(DATASET_DISPLAY.get(dataset, dataset))
        ax.set_xlabel("Threshold (Correct - Incorrect)")
        if ax == axes[0]:
            ax.set_ylabel("Percentage (%)")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 100)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        ordered = sorted(zip(labels, handles), key=lambda item: model_label_sort_key(item[0]))
        labels, handles = zip(*ordered)
        axes[-1].legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / THRESHOLD_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved threshold figure: {output_path}")


def score_labels(scores: pd.DataFrame) -> list[str]:
    """Return model labels present in the score table"""
    labels = score_label_series(scores)
    return labels.tolist()


def threshold_color_map(labels: list[str]) -> dict[str, str]:
    """Build the seeded color map used by threshold curves"""
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


if __name__ == "__main__":
    main()
