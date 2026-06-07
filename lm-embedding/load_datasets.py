"""Load lm-embedding evaluation data from MATCHA-prepared eval pickles."""

from __future__ import annotations

import ast
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetTable:
    """Normalized evaluation table used by embedding scorers."""

    name: str
    split: str
    frame: pd.DataFrame
    source_path: Path


PREPARED_EVAL_DATASETS: dict[str, tuple[str, str]] = {
    "snli": ("snli", "validation"),
    "multi_nli": ("multi_nli", "validation_matched"),
    "vitaminc": ("vitaminc", "validation"),
    "mednli": ("mednli", "validation"),
    "truthfulqa": ("truthfulqa", "validation"),
    "coco-caption": ("coco-caption", "validation"),
    "newts": ("newts", "validation"),
    "climate_fever": ("climate_fever", "validation"),
}

PAPER_PLOT_DATASETS = (
    "snli",
    "multi_nli",
    "truthfulqa",
    "climate_fever",
    "coco-caption",
    "newts",
)

DEFAULT_DATASETS = PAPER_PLOT_DATASETS

TRIPLET_COLUMNS = ("reference", "correct", "incorrect")
PREPARED_COLUMNS = {
    "premise": "reference",
    "correct_answer": "correct",
    "incorrect_answer": "incorrect",
}


def parse_dataset_specs(
    specs: list[str] | None,
    dataset_path: Path,
) -> list[DatasetTable]:
    """Load all dataset specs from a list of names or pickle paths."""
    requested = specs or list(DEFAULT_DATASETS)
    return [load_dataset_table(spec, dataset_path) for spec in requested]


def load_dataset_table(
    spec: str,
    dataset_path: Path,
) -> DatasetTable:
    """Load one dataset table from a prepared eval pickle.

    Supported spec forms:
      - dataset_name
      - dataset_name:split_name
      - dataset_name=/path/to/file.pkl
      - dataset_name=/path/to/file.pkl:split_name
    """
    name, path, split = parse_dataset_spec(spec, dataset_path)
    raw = load_pickle(path)

    if split is None:
        split = default_split_for(name)
    if split not in raw:
        available = ", ".join(raw.keys())
        raise KeyError(f"{path} does not contain split '{split}'. Available: {available}")

    frame = normalize_triplet_frame(raw[split], name)
    return DatasetTable(name=name, split=split, frame=frame, source_path=path)


def parse_dataset_spec(spec: str, dataset_path: Path) -> tuple[str, Path, str | None]:
    """Resolve a dataset name/path spec to canonical name, path, optional split."""
    if "=" in spec:
        raw_name, raw_target = spec.split("=", 1)
        name = canonical_dataset_name(raw_name)
        raw_path, split = split_optional_split(raw_target)
        path = Path(raw_path).expanduser()
    else:
        raw_name, split = split_optional_split(spec)
        path_candidate = Path(raw_name).expanduser()
        if path_candidate.suffix == ".pkl":
            path = path_candidate
            name = canonical_dataset_name(path.stem)
        else:
            name = canonical_dataset_name(raw_name)
            path = dataset_path / f"{name}.pkl"

    if not path.is_absolute():
        path = path.resolve()
    return name, path, split


def split_optional_split(value: str) -> tuple[str, str | None]:
    """Split a value into main text and an optional ':split' suffix."""
    if ":" not in value:
        return value, None
    main, maybe_split = value.rsplit(":", 1)
    if "/" in maybe_split or maybe_split.endswith(".pkl"):
        return value, None
    return main, maybe_split


def canonical_dataset_name(name: str) -> str:
    """Validate a prepare_eval_datasets.py dataset name."""
    key = name.strip().lower()
    if key not in PREPARED_EVAL_DATASETS:
        supported = ", ".join(sorted(PREPARED_EVAL_DATASETS))
        raise ValueError(f"Unsupported dataset '{name}'. Supported: {supported}")
    return PREPARED_EVAL_DATASETS[key][0]


def default_split_for(name: str) -> str:
    """Return the default evaluation split for a canonical dataset name."""
    for canonical, split in PREPARED_EVAL_DATASETS.values():
        if canonical == name:
            return split
    raise ValueError(f"No default split registered for '{name}'")


def load_pickle(path: Path) -> dict[str, pd.DataFrame]:
    """Read one prepared eval pickle and validate the outer structure."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prepared eval dataset: {path}. "
            "Run `python dataset_process/prepare_eval_datasets.py --data_path data` first."
        )

    with path.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a split-name dictionary")
    return data


def normalize_triplet_frame(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Convert prepared eval frames to reference/correct/incorrect text columns."""
    data = pd.DataFrame(frame).copy()
    missing = [col for col in PREPARED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing columns: {missing}")

    data = data.rename(columns=PREPARED_COLUMNS)
    data = data.loc[:, list(TRIPLET_COLUMNS)].reset_index(drop=True)

    for column in TRIPLET_COLUMNS:
        data[column] = data[column].map(extract_first_nonempty_text)

    return drop_empty_triplets(data, dataset_name)


def extract_first_nonempty_text(value: object) -> str:
    """Return a normalized scalar string from scalars, lists, arrays, or repr lists."""
    if is_missing_scalar(value):
        return ""

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        for item in value:
            text = extract_first_nonempty_text(item)
            if text:
                return text
        return ""

    text = normalize_text(value)
    if (text.startswith("[") and text.endswith("]")) or (
        text.startswith("(") and text.endswith(")")
    ):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
        return extract_first_nonempty_text(parsed)

    return text


def is_missing_scalar(value: object) -> bool:
    """Return whether a value is scalar NA-like."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_text(value: object) -> str:
    """Normalize whitespace and NA-like values to a plain string."""
    if value is None:
        return ""
    try:
        if isinstance(value, float) and np.isnan(value):
            return ""
    except TypeError:
        pass
    return " ".join(str(value).split())


def drop_empty_triplets(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Drop rows with an empty reference/correct/incorrect field."""
    mask = frame.apply(
        lambda row: any(not str(row[col]).strip() for col in TRIPLET_COLUMNS),
        axis=1,
    )
    dropped = int(mask.sum())
    if dropped:
        print(f"[{dataset_name}] dropped {dropped} rows with empty triplet fields")
    return frame.loc[~mask].reset_index(drop=True)
