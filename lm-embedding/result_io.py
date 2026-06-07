"""Read and write lm-embedding result tables."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


SCORES_LONG_FILENAME = "scores_long.csv"
SUMMARY_FILENAME = "summary.csv"
SUMMARY_ALL_FILENAME = "summary_all.csv"
MODEL_SCORES_DIRNAME = "model_scores"

MODEL_SCORE_COLUMNS = [
    "dataset",
    "kind",
    "row_id",
    "model",
    "model_display",
    "embedding_details",
    "reference",
    "correct",
    "incorrect",
    "correct_sim",
    "incorrect_sim",
    "gap",
]

SUMMARY_COLUMNS = [
    "dataset",
    "model",
    "model_display",
    "embedding_details",
    "n_rows",
    "correct_mean",
    "incorrect_mean",
    "gap_mean",
]


def model_scores_dir(output_dir: Path, dataset_name: str) -> Path:
    """Return the model-score directory for a dataset."""
    return output_dir / dataset_name / MODEL_SCORES_DIRNAME


def model_scores_path(output_dir: Path, dataset_name: str, model_key: str) -> Path:
    """Return the model-score CSV path for one dataset/model pair."""
    return model_scores_dir(output_dir, dataset_name) / f"{safe_filename(model_key)}.csv"


def write_model_scores(path: Path, scores: pd.DataFrame) -> None:
    """Write a per-model score table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scores = scores.loc[:, MODEL_SCORE_COLUMNS]
    scores.to_csv(path, index=False)
    print(f"Wrote {path}")


def load_model_scores(path: Path) -> pd.DataFrame:
    """Read a per-model score table."""
    return pd.read_csv(path)


def write_dataset_score_tables(output_dir: Path, dataset_name: str) -> pd.DataFrame:
    """Rebuild scores_long.csv and summary.csv from model_scores/*.csv."""
    score_dir = model_scores_dir(output_dir, dataset_name)
    frames = [pd.read_csv(path) for path in sorted(score_dir.glob("*.csv"))]
    if frames:
        scores_long = pd.concat(frames, ignore_index=True)
    else:
        scores_long = pd.DataFrame(columns=MODEL_SCORE_COLUMNS)

    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    scores_long.to_csv(dataset_dir / SCORES_LONG_FILENAME, index=False)

    summary = summarize_scores(scores_long)
    summary.to_csv(dataset_dir / SUMMARY_FILENAME, index=False)
    print(f"Wrote {dataset_dir / SCORES_LONG_FILENAME}")
    print(f"Wrote {dataset_dir / SUMMARY_FILENAME}")
    return summary


def write_all_datasets_summary(output_dir: Path) -> pd.DataFrame:
    """Rebuild summary_all.csv from all dataset summaries under output_dir."""
    summaries = [
        pd.read_csv(path)
        for path in sorted(output_dir.glob(f"*/{SUMMARY_FILENAME}"))
    ]
    if summaries:
        summary_all = pd.concat(summaries, ignore_index=True)
    else:
        summary_all = pd.DataFrame(columns=SUMMARY_COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_all.to_csv(output_dir / SUMMARY_ALL_FILENAME, index=False)
    print(f"Wrote {output_dir / SUMMARY_ALL_FILENAME}")
    return summary_all


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize model score rows by dataset/model metadata."""
    if scores.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

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

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def safe_filename(value: str) -> str:
    """Create a conservative filename from a model key."""
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", value).strip("_")
