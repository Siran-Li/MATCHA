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

PREPARED_EVAL_DATASETS = {
    # Keep this explicit, guessing column names can silently change a dataset split
    "snli": PREPARED_TRIPLET_SCHEMA,
    "multi_nli": PREPARED_TRIPLET_SCHEMA,
    "truthfulqa": PREPARED_TRIPLET_SCHEMA,
    "climate_fever": PREPARED_TRIPLET_SCHEMA,
    "coco-caption": PREPARED_TRIPLET_SCHEMA,
    "newts": PREPARED_TRIPLET_SCHEMA,
}

PICKLE_SUFFIXES = {".pkl", ".pickle"}


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
    if path.suffix.lower() not in PICKLE_SUFFIXES:
        raise ValueError(
            f"LM embedding expects prepared .pkl datasets, got {path.name}, "
            "Run dataset_process/prepare_eval_datasets.py first"
        )

    df = load_pickle_frame(path, split=split)
    schema = resolve_dataset_schema(name, path, df)
    frame = normalize_triplet_frame(df, schema)
    return DatasetTable(name=name, path=path, frame=frame)


def resolve_dataset_schema(name: str, path: Path, df: pd.DataFrame) -> TripletDatasetSchema:
    """Return the exact column mapping for a supported LM embedding dataset"""
    candidates = [normalize_dataset_key(name), normalize_dataset_key(path.stem)]
    for key in candidates:
        if key in PREPARED_EVAL_DATASETS:
            return PREPARED_EVAL_DATASETS[key]

    raise ValueError(
        f"No explicit dataset schema for '{name}' ({path.name}), "
        f"supported LM embedding datasets are: {', '.join(sorted(PREPARED_EVAL_DATASETS))}, "
        f"found columns: {list(df.columns)}"
    )


def normalize_dataset_key(name: str) -> str:
    """Normalize dataset names for registry lookup"""
    return name.strip().lower()


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


def normalize_triplet_frame(df: pd.DataFrame, schema: TripletDatasetSchema) -> pd.DataFrame:
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
    return drop_empty_triplets(out)


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


def drop_empty_triplets(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that cannot provide all three texts for scoring"""
    mask = (df["reference"].str.len() > 0) & (df["correct"].str.len() > 0) & (df["incorrect"].str.len() > 0)
    removed = int((~mask).sum())
    if removed:
        print(f"Filtered {removed} rows with empty reference/correct/incorrect text")
    return df[mask].reset_index(drop=True)
