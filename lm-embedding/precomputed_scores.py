"""Import external score files into the LM embedding result format"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from load_datasets import DatasetTable, extract_first_nonempty_text


PRECOMPUTED_DISPLAY_NAMES = {
    "matcha": "MATCHA",
    "snpmi": "SNPMI",
}


@dataclass(frozen=True)
class PrecomputedScoreSpec:
    """CLI spec for one imported score file"""

    dataset: str
    model: str
    path: Path

    @property
    def display_name(self) -> str:
        """Return the plot label for this imported score source"""
        return PRECOMPUTED_DISPLAY_NAMES.get(self.model, self.model)


@dataclass(frozen=True)
class PrecomputedColumnMapping:
    """Exact external column names for score import"""

    correct_sim: str
    incorrect_sim: str
    reference: str | None = None
    correct: str | None = None
    incorrect: str | None = None

    @property
    def text_columns(self) -> tuple[str, str, str] | None:
        """Return triplet text columns when the external file carries them"""
        if self.reference is None or self.correct is None or self.incorrect is None:
            return None
        return (self.reference, self.correct, self.incorrect)

    @property
    def required_score_columns(self) -> tuple[str, str]:
        """Return the two score columns needed for the experiment"""
        return (self.correct_sim, self.incorrect_sim)


# These mappings are exact on purpose, substring aliases caused too much ambiguity here
CANONICAL_SCORE_MAPPING = PrecomputedColumnMapping(
    reference="reference",
    correct="correct",
    incorrect="incorrect",
    correct_sim="correct_sim",
    incorrect_sim="incorrect_sim",
)

SCORE_ONLY_POS_NEG_MAPPING = PrecomputedColumnMapping(
    correct_sim="pos_sim",
    incorrect_sim="neg_sim",
)

MATCHA_EVAL_SCORE_MAPPING = PrecomputedColumnMapping(
    reference="premise",
    correct="correct_answer",
    incorrect="incorrect_answer",
    correct_sim="pos_sim",
    incorrect_sim="neg_sim",
)

PRECOMPUTED_COLUMN_MAPPINGS = {
    "matcha": MATCHA_EVAL_SCORE_MAPPING,
    "snpmi": SCORE_ONLY_POS_NEG_MAPPING,
}


def parse_precomputed_score_specs(score_args: list[str]) -> dict[str, list[PrecomputedScoreSpec]]:
    """Parse repeated --precomputed-score DATASET:MODEL=PATH arguments"""
    grouped: dict[str, list[PrecomputedScoreSpec]] = {}
    for item in score_args:
        spec = parse_precomputed_score_spec(item)
        grouped.setdefault(spec.dataset, []).append(spec)
    return grouped


def parse_precomputed_score_spec(item: str) -> PrecomputedScoreSpec:
    """Parse one external score argument"""
    if "=" not in item or ":" not in item.split("=", 1)[0]:
        raise ValueError(f"--precomputed-score must be DATASET:MODEL=PATH, got: {item}")
    left, path = item.split("=", 1)
    dataset, model = left.split(":", 1)
    dataset = dataset.strip()
    model = model.strip().lower()
    path = path.strip()
    if not dataset or not model or not path:
        raise ValueError(f"--precomputed-score must be DATASET:MODEL=PATH, got: {item}")
    return PrecomputedScoreSpec(dataset=dataset, model=model, path=Path(path).expanduser())


def load_precomputed_scores(
    table: DatasetTable,
    spec: PrecomputedScoreSpec,
    denormalize: bool = False,
) -> pd.DataFrame:
    """Load external pos/neg scores and align them to the prepared dataset"""
    if not spec.path.exists():
        raise FileNotFoundError(f"Precomputed score file not found: {spec.path}")

    mapping = precomputed_column_mapping(spec.model)
    score_df = load_precomputed_score_frame(spec.path, mapping)
    validate_precomputed_columns(score_df, spec, mapping)

    aligned = align_score_frame(table, score_df, mapping)
    correct_sim = pd.to_numeric(aligned[mapping.correct_sim], errors="coerce")
    incorrect_sim = pd.to_numeric(aligned[mapping.incorrect_sim], errors="coerce")
    if denormalize:
        # Only apply this for imported scores that were saved on a unit interval
        correct_sim = denormalize_unit_interval_scores(correct_sim)
        incorrect_sim = denormalize_unit_interval_scores(incorrect_sim)

    df = table.frame.reset_index(drop=True)
    return pd.DataFrame(
        {
            "dataset": table.name,
            "kind": "triplet",
            "row_id": df["row_id"].to_numpy(),
            "model": spec.model,
            "model_display": spec.display_name,
            "embedding_details": spec.model,
            "reference": df["reference"].to_numpy(),
            "correct": df["correct"].to_numpy(),
            "incorrect": df["incorrect"].to_numpy(),
            "correct_sim": correct_sim.to_numpy(),
            "incorrect_sim": incorrect_sim.to_numpy(),
            "gap": correct_sim.to_numpy() - incorrect_sim.to_numpy(),
        }
    )


def denormalize_unit_interval_scores(scores: pd.Series) -> pd.Series:
    """Map scores from [0, 1] to [-1, 1]"""
    return (2 * scores) - 1


def load_precomputed_score_frame(path: Path, mapping: PrecomputedColumnMapping) -> pd.DataFrame:
    """Load precomputed scores from CSV or pickle into a DataFrame"""
    if path.suffix.lower() in {".pkl", ".pickle"}:
        data = pd.read_pickle(path)
    else:
        data = pd.read_csv(path)

    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        # Some pickle exports are dicts, only treat them as score tables if score keys exist
        columns = list(mapping.required_score_columns)
        text_columns = mapping.text_columns
        if text_columns is not None:
            columns.extend(text_columns)
        if "row_id" in data:
            columns.append("row_id")
        present = {column: data[column] for column in columns if column in data}
        missing = [column for column in mapping.required_score_columns if column not in present]
        if missing:
            raise ValueError(
                f"Precomputed score pickle {path} is a dict but is missing required score keys: {missing}"
            )
        return pd.DataFrame(present)
    raise ValueError(f"Unsupported precomputed score object in {path}: {type(data).__name__}")


def precomputed_column_mapping(model: str) -> PrecomputedColumnMapping:
    """Return the exact expected CSV schema for a precomputed score model"""
    return PRECOMPUTED_COLUMN_MAPPINGS.get(model, CANONICAL_SCORE_MAPPING)


def validate_precomputed_columns(
    df: pd.DataFrame,
    spec: PrecomputedScoreSpec,
    mapping: PrecomputedColumnMapping,
) -> None:
    """Fail early when a precomputed file is not in the expected exact schema"""
    missing = [column for column in mapping.required_score_columns if column not in df.columns]
    if missing:
        expected = list(mapping.required_score_columns)
        raise ValueError(
            f"Precomputed score file for {spec.model} has unexpected columns, "
            f"missing required score columns: {missing}, expected score schema: {expected}, "
            f"Found columns: {list(df.columns)}"
        )


def align_score_frame(
    table: DatasetTable,
    score_df: pd.DataFrame,
    mapping: PrecomputedColumnMapping,
) -> pd.DataFrame:
    """Align external scores by triplet text, row_id, or row order"""
    # Prefer real alignment keys, row order is only used for score-only exports
    text_columns = mapping.text_columns
    if text_columns is not None and all(column in score_df.columns for column in text_columns):
        return align_by_triplet_text(table, score_df, text_columns)
    if "row_id" in score_df.columns:
        return align_by_row_id(table, score_df)
    return align_by_row_order(table, score_df)


def align_by_triplet_text(
    table: DatasetTable,
    score_df: pd.DataFrame,
    text_columns: tuple[str, str, str],
) -> pd.DataFrame:
    """Align scores to the prepared table using normalized reference/correct/incorrect text"""
    ref_col, correct_col, incorrect_col = text_columns
    scores = score_df.copy()
    scores["_join_key"] = make_join_key(scores[ref_col], scores[correct_col], scores[incorrect_col])
    scores = scores.drop_duplicates("_join_key", keep="first")

    lookup = table.frame[["reference", "correct", "incorrect"]].copy()
    lookup["_join_key"] = make_join_key(lookup["reference"], lookup["correct"], lookup["incorrect"])
    aligned = lookup[["_join_key"]].merge(scores, on="_join_key", how="left")
    missing_rows = aligned.drop(columns=["_join_key"]).isna().all(axis=1)
    if missing_rows.any():
        missing = int(missing_rows.sum())
        raise ValueError(f"Precomputed scores are missing {missing} rows after text alignment")
    return aligned.drop(columns=["_join_key"])


def align_by_row_id(table: DatasetTable, score_df: pd.DataFrame) -> pd.DataFrame:
    """Align scores using row_id when the external file already has prepared row ids"""
    lookup = table.frame[["row_id"]]
    aligned = lookup.merge(score_df, on="row_id", how="left")
    if len(aligned) != len(table.frame):
        raise ValueError("Precomputed row_id alignment changed the number of rows")
    missing_rows = aligned.drop(columns=["row_id"]).isna().all(axis=1)
    if missing_rows.any():
        missing = int(missing_rows.sum())
        raise ValueError(f"Precomputed scores are missing {missing} rows after row_id alignment")
    return aligned


def align_by_row_order(table: DatasetTable, score_df: pd.DataFrame) -> pd.DataFrame:
    """Align scores by row order when no safer key is available"""
    if len(score_df) != len(table.frame):
        raise ValueError(
            f"Precomputed scores have {len(score_df)} rows, but dataset {table.name} has {len(table.frame)} rows, "
            "Provide row_id or triplet text columns to align safely"
        )
    return score_df.reset_index(drop=True)


def make_join_key(reference: pd.Series, correct: pd.Series, incorrect: pd.Series) -> pd.Series:
    """Build normalized triplet keys for text-based score alignment"""
    return pd.Series(
        zip(
            reference.map(extract_first_nonempty_text),
            correct.map(extract_first_nonempty_text),
            incorrect.map(extract_first_nonempty_text),
        ),
        index=reference.index,
    )


def summarize_precomputed_scores(records: pd.DataFrame, spec: PrecomputedScoreSpec) -> dict[str, object]:
    """Create one summary row for an external score file"""
    correct_mean = float(records["correct_sim"].mean(skipna=True))
    incorrect_mean = float(records["incorrect_sim"].mean(skipna=True))
    return {
        "dataset": spec.dataset,
        "model": spec.model,
        "model_display": spec.display_name,
        "embedding_details": spec.model,
        "n_rows": int(len(records)),
        "correct_mean": correct_mean,
        "incorrect_mean": incorrect_mean,
        "gap_mean": correct_mean - incorrect_mean,
    }
