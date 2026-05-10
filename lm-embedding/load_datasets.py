"""Load prepared MATCHA evaluation datasets for LM embedding scoring"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from load_models import normalize_text


@dataclass
class DatasetTable:
    """Prepared scoring table plus its source path"""

    name: str
    path: Path
    frame: pd.DataFrame
    allow_partial_answers: bool = False


@dataclass(frozen=True)
class TripletDatasetSchema:
    """Exact source columns for one reference/correct/incorrect dataset"""

    reference: str
    correct: str
    incorrect: str


PREPARED_TRIPLET_SCHEMA = TripletDatasetSchema(
    reference="premise",
    correct="correct_answer",
    incorrect="incorrect_answer",
)

CSV_TRIPLET_SCHEMA = TripletDatasetSchema(
    reference="premise",
    correct="correct",
    incorrect="incorrect",
)

PREPARED_EVAL_DATASETS = {
    # Keep this explicit, guessing column names can silently change a dataset split
    "snli": PREPARED_TRIPLET_SCHEMA,
    "multi_nli": PREPARED_TRIPLET_SCHEMA,
    "truthfulqa": PREPARED_TRIPLET_SCHEMA,
    "climate_fever": PREPARED_TRIPLET_SCHEMA,
    "coco-caption": PREPARED_TRIPLET_SCHEMA,
    "newts": PREPARED_TRIPLET_SCHEMA,
}

CSV_EVAL_DATASETS = {
    "snli": CSV_TRIPLET_SCHEMA,
    "multi_nli": CSV_TRIPLET_SCHEMA,
    "truthfulqa": CSV_TRIPLET_SCHEMA,
    "truthfulqa_filtered": CSV_TRIPLET_SCHEMA,
    "climate_fever": CSV_TRIPLET_SCHEMA,
    "climate_fever_150": CSV_TRIPLET_SCHEMA,
    "coco-caption": CSV_TRIPLET_SCHEMA,
    "coco-caption-concat": CSV_TRIPLET_SCHEMA,
    "newts": CSV_TRIPLET_SCHEMA,
    "newts_random_first1sent": CSV_TRIPLET_SCHEMA,
}

PICKLE_SUFFIXES = {".pkl", ".pickle"}
CSV_SUFFIXES = {".csv"}
SUPPORTED_SUFFIXES = PICKLE_SUFFIXES | CSV_SUFFIXES


def parse_dataset_specs(dataset_args: list[str]) -> list[tuple[str, Path]]:
    """Parse repeated --dataset NAME=PATH arguments"""
    if not dataset_args:
        raise ValueError("Provide at least one dataset as --dataset NAME=PATH")
    specs = []
    for item in dataset_args:
        specs.append(parse_dataset_spec(item))
    return specs


def parse_dataset_spec(item: str) -> tuple[str, Path]:
    """Parse one dataset argument into a dataset name and local path"""
    if "=" not in item:
        raise ValueError(f"--dataset must be NAME=PATH, got: {item}")
    name, path = item.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"--dataset must be NAME=PATH, got: {item}")
    return name, Path(path).expanduser()


def load_dataset_table(name: str, path: Path, split: str | None = None) -> DatasetTable:
    """Load one prepared dataset and convert it to the scoring table format"""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"LM embedding expects prepared .csv or .pkl datasets, got {path.name}"
        )

    if suffix in PICKLE_SUFFIXES:
        df = load_pickle_frame(path, split=split)
    else:
        if split is not None:
            raise ValueError(f"--split is only supported for pickle datasets, got CSV: {path}")
        df = pd.read_csv(path)

    schema = resolve_dataset_schema(name, path, df)
    allow_partial_answers = suffix in CSV_SUFFIXES
    frame = normalize_triplet_frame(df, schema, allow_partial_answers=allow_partial_answers)
    return DatasetTable(
        name=name,
        path=path,
        frame=frame,
        allow_partial_answers=allow_partial_answers,
    )


def resolve_dataset_schema(name: str, path: Path, df: pd.DataFrame) -> TripletDatasetSchema:
    """Return the exact column mapping for a supported LM embedding dataset"""
    candidates = [normalize_dataset_key(name), normalize_dataset_key(path.stem)]
    for key in candidates:
        for registry in (PREPARED_EVAL_DATASETS, CSV_EVAL_DATASETS):
            schema = registry.get(key)
            if schema is not None and has_columns(df, [schema.reference, schema.correct, schema.incorrect]):
                return schema

    raise ValueError(
        f"No explicit dataset schema for '{name}' ({path.name}), "
        f"supported LM embedding datasets are: {', '.join(sorted(PREPARED_EVAL_DATASETS))}, "
        f"found columns: {list(df.columns)}"
    )


def normalize_dataset_key(name: str) -> str:
    """Normalize dataset names for registry lookup"""
    return name.strip().lower()


def has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    """Return whether all schema columns are present"""
    return all(column in df.columns for column in columns)


def load_pickle_frame(path: Path, split: str | None = None) -> pd.DataFrame:
    """Load a prepared pickle and select the requested or default split"""
    with open(path, "rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, dict):
        # Prepared pickles usually store split names at the top level
        if split is None:
            split = next((key for key in ["validation", "validation_matched", "test", "val"] if key in obj), None)
        if split is None:
            split = next(iter(obj))
        obj = obj[split]
    if not isinstance(obj, pd.DataFrame):
        obj = pd.DataFrame(obj)
    return obj.reset_index(drop=True)


def normalize_triplet_frame(
    df: pd.DataFrame,
    schema: TripletDatasetSchema,
    allow_partial_answers: bool = False,
) -> pd.DataFrame:
    """Rename prepared triplet columns to reference/correct/incorrect"""
    require_columns(df, [schema.reference, schema.correct, schema.incorrect])

    out = pd.DataFrame(
        {
            "row_id": np.arange(len(df)),
            "reference": df[schema.reference].map(extract_first_nonempty_text),
            "correct": df[schema.correct].map(extract_first_nonempty_text),
            "incorrect": df[schema.incorrect].map(extract_first_nonempty_text),
        }
    )
    return drop_empty_triplets(out, allow_partial_answers=allow_partial_answers)


def require_columns(df: pd.DataFrame, columns: list[str | None]) -> None:
    """Raise a clear error when required columns are missing"""
    required = [column for column in columns if column is not None]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}, found columns: {list(df.columns)}")


def extract_first_nonempty_text(value: object) -> str:
    """Return the first nonempty text, missing cells become empty and are filtered"""
    if is_missing_scalar(value):
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        # Some prepared answer fields are lists, the scorer expects one text string
        for item in value:
            if is_missing_scalar(item):
                continue
            text = normalize_text(item)
            if text:
                return text
        return ""
    text = normalize_text(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return extract_first_nonempty_text(parsed)
        except (SyntaxError, ValueError):
            return text
    return text


def is_missing_scalar(value: object) -> bool:
    """Return True for scalar missing values such as None or NaN"""
    if isinstance(value, (list, tuple, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def drop_empty_triplets(df: pd.DataFrame, allow_partial_answers: bool = False) -> pd.DataFrame:
    """Remove rows that cannot provide enough text for scoring"""
    reference_mask = df["reference"].str.len() > 0
    if allow_partial_answers:
        answer_mask = (df["correct"].str.len() > 0) | (df["incorrect"].str.len() > 0)
    else:
        answer_mask = (df["correct"].str.len() > 0) & (df["incorrect"].str.len() > 0)
    mask = reference_mask & answer_mask
    removed = int((~mask).sum())
    if removed:
        print(f"Filtered {removed} rows with empty reference/correct/incorrect text")
    return df[mask].reset_index(drop=True)
