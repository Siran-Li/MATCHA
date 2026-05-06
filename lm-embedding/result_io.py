from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from load_models import ModelSpec, display_name, embedding_details


SCORES_LONG_FILENAME = "scores_long.csv"
SUMMARY_FILENAME = "summary.csv"
SUMMARY_ALL_FILENAME = "summary_all.csv"
MODEL_SCORES_DIRNAME = "model_scores"

# Keep the CSV layout stable so cached scores, plotting, and summaries
# can all read the same files without guessing column order
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


def model_scores_dir(output_dir: Path, dataset_name: str) -> Path:
    return output_dir / dataset_name / MODEL_SCORES_DIRNAME


def model_scores_path(output_dir: Path, dataset_name: str, spec: ModelSpec) -> Path:
    filename = f"{safe_filename(embedding_details(spec))}.csv"
    return model_scores_dir(output_dir, dataset_name) / filename


def write_model_scores(records: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Older/imported score tables may not have every metadata column
    # Write the known columns in order and ignore anything missing
    columns = [column for column in MODEL_SCORE_COLUMNS if column in records.columns]

    records.to_csv(path, index=False, columns=columns)
    print(f"Saved model scores: {path}")


def load_model_scores(path: Path, dataset_name: str, spec: ModelSpec) -> pd.DataFrame:
    records = pd.read_csv(path)
    return fill_missing_score_metadata(records, dataset_name, spec)


def fill_missing_score_metadata(records: pd.DataFrame, dataset_name: str, spec: ModelSpec) -> pd.DataFrame:
    # This lets --skip-existing reuse score files from earlier runs,
    # even if those files were written before all metadata columns existed.
    if "dataset" not in records.columns:
        records["dataset"] = dataset_name
    if "model" not in records.columns:
        records["model"] = spec.key
    if "model_display" not in records.columns:
        records["model_display"] = display_name(spec.key)
    if "embedding_details" not in records.columns:
        records["embedding_details"] = embedding_details(spec)
    if "gap" not in records.columns and {"correct_sim", "incorrect_sim"}.issubset(records.columns):
        records["gap"] = records["correct_sim"] - records["incorrect_sim"]
    return records


def write_dataset_score_tables(
    output_dir: Path,
    dataset_name: str,
    records_by_model: list[pd.DataFrame],
    summaries: list[dict[str, object]],
) -> None:
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if records_by_model:
        # The long file is the main plotting input
        scores_path = dataset_dir / SCORES_LONG_FILENAME
        pd.concat(records_by_model, ignore_index=True).to_csv(scores_path, index=False)
        print(f"Saved long scores: {scores_path}")

    if summaries:
        summary_path = dataset_dir / SUMMARY_FILENAME
        pd.DataFrame(summaries).to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")


def write_all_datasets_summary(output_dir: Path, summary_tables: list[pd.DataFrame]) -> None:
    if not summary_tables:
        return

    path = output_dir / SUMMARY_ALL_FILENAME
    pd.concat(summary_tables, ignore_index=True).to_csv(path, index=False)
    print(f"\nSaved combined summary: {path}")


def safe_filename(name: str) -> str:
    # Model names can contain slashes from Hugging Face IDs
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")