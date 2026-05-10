"""Compute semantic-separation scores from embedding encoders"""

from __future__ import annotations

import numpy as np
import pandas as pd

from load_datasets import DatasetTable
from load_models import EmbeddingEncoder, ModelSpec, display_name, embedding_details


def score_dataset_table(
    table: DatasetTable,
    spec: ModelSpec,
    encoder: EmbeddingEncoder,
    batch_size: int,
) -> pd.DataFrame:
    """Score a prepared reference/correct/incorrect triplet table"""
    return score_triplet_rows(table, spec, encoder, batch_size=batch_size)


def score_triplet_rows(
    table: DatasetTable,
    spec: ModelSpec,
    encoder: EmbeddingEncoder,
    batch_size: int,
) -> pd.DataFrame:
    """Score rows with reference/correct/incorrect text in the same record"""
    df = table.frame
    # Keep these as three aligned batches, batch size changes speed not row order
    reference_embeddings = encoder.encode(df["reference"].tolist(), batch_size=batch_size)
    validate_embedding_count(reference_embeddings, len(df), "reference", table.name, spec.key)

    # Compare the same reference against the two answer choices
    correct_sim = score_optional_answer_column(
        df,
        reference_embeddings,
        "correct",
        encoder,
        batch_size,
        table.name,
        spec.key,
    )
    incorrect_sim = score_optional_answer_column(
        df,
        reference_embeddings,
        "incorrect",
        encoder,
        batch_size,
        table.name,
        spec.key,
    )

    return pd.DataFrame(
        {
            **score_metadata(table, spec),
            "row_id": df["row_id"].to_numpy(),
            "reference": df["reference"].to_numpy(),
            "correct": df["correct"].to_numpy(),
            "incorrect": df["incorrect"].to_numpy(),
            "correct_sim": correct_sim,
            "incorrect_sim": incorrect_sim,
            "gap": correct_sim - incorrect_sim,
        }
    )


def score_optional_answer_column(
    df: pd.DataFrame,
    reference_embeddings: np.ndarray,
    column: str,
    encoder: EmbeddingEncoder,
    batch_size: int,
    dataset: str,
    model: str,
) -> np.ndarray:
    """Score a candidate column while preserving NaNs for missing one-sided CSV rows."""
    scores = np.full(len(df), np.nan, dtype=np.float32)
    mask = df[column].str.len() > 0
    if not mask.any():
        return scores

    answer_embeddings = encoder.encode(df.loc[mask, column].tolist(), batch_size=batch_size)
    validate_embedding_count(answer_embeddings, int(mask.sum()), column, dataset, model)
    scores[mask.to_numpy()] = cosine_similarity_by_row(reference_embeddings[mask.to_numpy()], answer_embeddings)
    return scores


def score_metadata(table: DatasetTable, spec: ModelSpec) -> dict[str, object]:
    """Shared metadata copied onto every scored row"""
    return {
        "dataset": table.name,
        "kind": "triplet",
        "model": spec.key,
        "model_display": display_name(spec.key),
        "embedding_details": embedding_details(spec),
    }


def cosine_similarity_by_row(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return cosine similarity for each aligned pair of embedding rows"""
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    numerator = np.sum(left * right, axis=1)
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    scores = numerator / np.where(denom == 0, np.nan, denom)
    scores[~np.isfinite(scores)] = np.nan
    return scores


def validate_embedding_count(
    embeddings: np.ndarray,
    expected_rows: int,
    column: str,
    dataset: str,
    model: str,
) -> None:
    """Fail loudly if an encoder loses row alignment."""
    actual_rows = len(embeddings)
    if actual_rows != expected_rows:
        raise ValueError(
            f"Encoder returned {actual_rows} {column} embeddings for {expected_rows} "
            f"rows on {dataset}/{model}; row alignment would be invalid"
        )


def drop_invalid_score_rows(
    records: pd.DataFrame,
    keep_invalid: bool,
    allow_partial_scores: bool = False,
) -> pd.DataFrame:
    """Drop rows where either candidate similarity could not be computed"""
    if keep_invalid:
        return records
    # Gap comes from these two columns, so checking both is enough
    required_columns = ["correct_sim", "incorrect_sim"]
    if allow_partial_scores:
        mask = records[required_columns].notna().any(axis=1)
    else:
        mask = records[required_columns].notna().all(axis=1)
    removed = int((~mask).sum())
    if removed:
        print(f"Dropped {removed} rows with invalid embeddings")
    return records[mask].reset_index(drop=True)


def summarize_model_scores(records: pd.DataFrame, dataset: str, spec: ModelSpec) -> dict[str, object]:
    """Create one summary row for the model/dataset pair"""
    correct_mean = float(records["correct_sim"].mean(skipna=True))
    incorrect_mean = float(records["incorrect_sim"].mean(skipna=True))
    return {
        "dataset": dataset,
        "model": spec.key,
        "model_display": display_name(spec.key),
        "embedding_details": embedding_details(spec),
        "n_rows": int(len(records)),
        "correct_mean": correct_mean,
        "incorrect_mean": incorrect_mean,
        "gap_mean": correct_mean - incorrect_mean,
    }
