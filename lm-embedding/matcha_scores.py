"""Run eval_matcha.py and import MATCHA results into lm-embedding tables."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from result_io import (
    model_scores_path,
    write_all_datasets_summary,
    write_dataset_score_tables,
    write_model_scores,
)


MATCHA_MODEL_KEY = "matcha"
MATCHA_DISPLAY_NAME = "MATCHA"
MATCHA_EVAL_DATASETS = {
    "snli",
    "multi_nli",
    "mednli",
    "truthfulqa",
    "coco-caption",
    "newts",
    "climate_fever",
}


def run_matcha_eval(
    *,
    matcha_output_path: Path,
    dataset_path: Path,
    datasets: list[str],
    tag: str,
    model_name: str,
    python_executable: Path | None = None,
) -> Path:
    """Execute MATCHA evaluation through the repository's eval_matcha.py."""
    unsupported = sorted(set(datasets) - MATCHA_EVAL_DATASETS)
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(f"eval_matcha.py does not currently support: {joined}")

    matcha_output_path = matcha_output_path.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    matcha_root = Path(__file__).resolve().parents[1]
    eval_script = matcha_root / "eval_matcha.py"
    python_cmd = str(python_executable or Path(sys.executable))
    results_dir = matcha_output_path / tag

    command = [
        python_cmd,
        str(eval_script),
        "--output-path",
        str(matcha_output_path),
        "--model-name",
        model_name,
        "--dataset-path",
        str(dataset_path),
        "--tag",
        tag,
        "--datasets",
        *datasets,
    ]
    print("Running MATCHA:", " ".join(command))
    subprocess.run(command, cwd=matcha_root, check=True)
    return results_dir


def import_matcha_results(
    *,
    results_dir: Path,
    output_dir: Path,
    datasets: list[str],
) -> None:
    """Import eval_matcha.py CSV outputs into lm-embedding result tables."""
    imported: set[str] = set()
    for dataset in datasets:
        result_path = results_dir / f"{dataset}_results.csv"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing MATCHA result CSV: {result_path}")

        raw = pd.read_csv(result_path)
        scores = convert_matcha_result_frame(dataset, raw)
        write_model_scores(model_scores_path(output_dir, dataset, MATCHA_MODEL_KEY), scores)
        write_dataset_score_tables(output_dir, dataset)
        imported.add(dataset)

    if imported:
        write_all_datasets_summary(output_dir)


def convert_matcha_result_frame(dataset: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Convert one eval_matcha.py result CSV to the lm-embedding score schema."""
    if {"premise", "correct_answer", "incorrect_answer", "pos_sim", "neg_sim"}.issubset(raw.columns):
        return convert_triplet_result_frame(dataset, raw)
    if {"text1", "text2", "label", "score"}.issubset(raw.columns):
        return convert_pairwise_result_frame(dataset, raw)

    raise ValueError(
        f"{dataset} MATCHA results have an unknown schema. "
        f"Columns: {list(raw.columns)}"
    )


def convert_triplet_result_frame(dataset: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Convert triplet MATCHA results to the lm-embedding score schema."""
    data = raw.reset_index(drop=True).copy()
    scores = pd.DataFrame({
        "dataset": dataset,
        "kind": "triplet",
        "row_id": np.arange(len(data)),
        "model": MATCHA_MODEL_KEY,
        "model_display": MATCHA_DISPLAY_NAME,
        "embedding_details": MATCHA_MODEL_KEY,
        "reference": data["premise"],
        "correct": data["correct_answer"],
        "incorrect": data["incorrect_answer"],
        "correct_sim": data["pos_sim"],
        "incorrect_sim": data["neg_sim"],
    })
    scores["gap"] = scores["correct_sim"] - scores["incorrect_sim"]
    return scores


def convert_pairwise_result_frame(dataset: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Convert pairwise MATCHA results to the lm-embedding score schema."""
    data = raw.reset_index(drop=True).copy()
    labels = data["label"].astype(int)
    scores = pd.DataFrame({
        "dataset": dataset,
        "kind": "pairwise",
        "row_id": np.arange(len(data)),
        "model": MATCHA_MODEL_KEY,
        "model_display": MATCHA_DISPLAY_NAME,
        "embedding_details": MATCHA_MODEL_KEY,
        "reference": data["text1"],
        "correct": data["text2"].where(labels == 1, ""),
        "incorrect": data["text2"].where(labels == 0, ""),
        "correct_sim": data["score"].where(labels == 1, np.nan),
        "incorrect_sim": data["score"].where(labels == 0, np.nan),
        "gap": np.nan,
    })
    return scores
