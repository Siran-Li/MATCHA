"""Compute row-level embedding similarities for prepared triplet datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from load_datasets import DatasetTable
from load_models import EmbeddingEncoder, ModelSpec, display_name, embedding_details


def score_dataset_table(
    table: DatasetTable,
    encoder: EmbeddingEncoder,
    spec: ModelSpec,
) -> pd.DataFrame:
    """Compute positive/negative cosine scores for a dataset table."""
    frame = table.frame.reset_index(drop=True)
    reference_embeddings = encoder.encode(frame["reference"].tolist())
    correct_embeddings = encoder.encode(frame["correct"].tolist())
    incorrect_embeddings = encoder.encode(frame["incorrect"].tolist())

    validate_embedding_count(table.name, "reference", len(frame), reference_embeddings)
    validate_embedding_count(table.name, "correct", len(frame), correct_embeddings)
    validate_embedding_count(table.name, "incorrect", len(frame), incorrect_embeddings)

    correct_sim = cosine_similarity_by_row(reference_embeddings, correct_embeddings)
    incorrect_sim = cosine_similarity_by_row(reference_embeddings, incorrect_embeddings)

    scores = pd.DataFrame(
        {
            "dataset": table.name,
            "kind": "triplet",
            "row_id": np.arange(len(frame)),
            "model": spec.key,
            "model_display": display_name(spec),
            "embedding_details": embedding_details(spec),
            "reference": frame["reference"],
            "correct": frame["correct"],
            "incorrect": frame["incorrect"],
            "correct_sim": correct_sim,
            "incorrect_sim": incorrect_sim,
        }
    )
    scores["gap"] = scores["correct_sim"] - scores["incorrect_sim"]
    return scores


def cosine_similarity_by_row(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute row-wise cosine similarity and return NaN for invalid rows."""
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = numerator / denominator
    scores[~np.isfinite(scores)] = np.nan
    return scores


def validate_embedding_count(
    dataset_name: str,
    column_name: str,
    expected_count: int,
    embeddings: np.ndarray,
) -> None:
    """Raise if a model returned a different number of embeddings than inputs."""
    actual_count = len(embeddings)
    if actual_count != expected_count:
        raise ValueError(
            f"{dataset_name}/{column_name}: expected {expected_count} embeddings, got {actual_count}"
        )


def drop_invalid_score_rows(
    scores: pd.DataFrame,
    dataset_name: str,
    *,
    keep_invalid: bool = False,
) -> pd.DataFrame:
    """Drop rows with missing similarity scores unless requested otherwise."""
    if keep_invalid:
        return scores

    required = ["correct_sim", "incorrect_sim"]
    mask = scores[required].notna().all(axis=1)
    dropped = int((~mask).sum())
    if dropped:
        print(f"[{dataset_name}] dropped {dropped} rows with invalid scores")
    return scores.loc[mask].reset_index(drop=True)


def summarize_model_scores(scores: pd.DataFrame) -> dict[str, object]:
    """Create one summary row from a model score table."""
    return {
        "dataset": scores["dataset"].iloc[0],
        "model": scores["model"].iloc[0],
        "model_display": scores["model_display"].iloc[0],
        "embedding_details": scores["embedding_details"].iloc[0],
        "n_rows": int(len(scores)),
        "correct_mean": float(scores["correct_sim"].mean()),
        "incorrect_mean": float(scores["incorrect_sim"].mean()),
        "gap_mean": float(scores["gap"].mean()),
    }

