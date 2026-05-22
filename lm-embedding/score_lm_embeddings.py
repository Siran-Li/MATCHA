"""Run LM embedding scoring and save plot-ready CSV outputs"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys

import pandas as pd

from load_datasets import DatasetTable, load_dataset_table, parse_dataset_specs
from load_models import EmbeddingEncoder, available_model_keys, resolve_model_specs
from precomputed_scores import load_precomputed_scores, parse_precomputed_score_specs, summarize_precomputed_scores
from result_io import (
    load_model_scores,
    model_scores_dir,
    model_scores_path,
    safe_filename,
    write_all_datasets_summary,
    write_dataset_score_tables,
    write_model_scores,
)
from compute_scores import drop_invalid_score_rows, score_dataset_table, summarize_model_scores


GATED_MODEL_KEYS = {"mistral-7b", "llama-2-13b", "llama-3.1-8B-Instruct"}
FAILURES_FILENAME = "failures.csv"


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for dataset scoring and score import"""
    parser = argparse.ArgumentParser(description="Score LM embeddings and save CSVs for plotting")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Dataset input, can be repeated, example: --dataset snli=datasets/snli.csv",
    )
    parser.add_argument("--models", nargs="*", default=available_model_keys(), help=f"Embedding model keys, choices: {available_model_keys()}")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lm_embeddings"), help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--pooling-type", choices=["mean", "max", "pooler"], default="mean")
    parser.add_argument("--device", default=None, help="Torch device override, for example cuda:0 or cpu")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token, defaults to HF_TOKEN env var when omitted")
    parser.add_argument("--split", default=None, help="Split key to use for pkl files, default chooses validation/test-like split")
    parser.add_argument(
        "--precomputed-score",
        action="append",
        default=[],
        metavar="DATASET:MODEL=PATH",
        help="External score CSV to include without computing embeddings, example: --precomputed-score snli:matcha=outputs/snli_results.csv",
    )
    parser.add_argument(
        "--denormalize-precomputed-scores",
        action="store_true",
        help="Convert imported precomputed scores from [0, 1] to [-1, 1] using 2*x - 1",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip per-model CSVs that already exist")
    parser.add_argument("--keep-invalid", action="store_true", help="Keep rows whose embeddings are NaN instead of dropping them")
    parser.add_argument(
        "--ensure-nltk-data",
        action="store_true",
        help="Download missing NLTK tokenizer data needed by Word2Vec/GloVe",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first model/import failure instead of continuing",
    )
    return parser.parse_args()


def main() -> None:
    """Run the scorer across requested datasets, models, and imported scores"""
    args = parse_args()
    if args.hf_token:
        # Keep model-loading code reading from the normal Hugging Face env var
        os.environ["HF_TOKEN"] = args.hf_token

    dataset_specs = parse_dataset_specs(args.dataset)
    model_specs = apply_transformer_pooling(resolve_model_specs(args.models), args.pooling_type)
    warn_missing_hf_token(model_specs)
    precomputed_specs = parse_precomputed_score_specs(args.precomputed_score)
    validate_precomputed_datasets(dataset_specs, precomputed_specs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    failures = []
    for dataset_name, dataset_path in dataset_specs:
        table = load_dataset_table(dataset_name, dataset_path, split=args.split)
        print_dataset_header(table.name, len(table.frame))

        dataset_records = []
        summary_records = []
        for spec in model_specs:
            if missing_gated_model_token(spec):
                message = (
                    f"HF_TOKEN is required for gated Hugging Face model '{spec.key}'. "
                    "Export HF_TOKEN=hf_... or pass --hf-token to score this model."
                )
                failure = record_failure(table.name, spec.key, "model", "MissingHFToken", message)
                failures.append(failure)
                print_failure(failure)
                if args.fail_fast:
                    raise RuntimeError(message)
                continue

            try:
                records = load_or_compute_model_scores(args, table, spec)
            except Exception as exc:
                failure = record_failure_from_exception(table.name, spec.key, "model", exc)
                failures.append(failure)
                print_failure(failure)
                if args.fail_fast:
                    raise
                continue
            dataset_records.append(records)
            summary_records.append(summarize_model_scores(records, table.name, spec))

        for spec in precomputed_specs.get(table.name, []):
            try:
                records = load_and_write_precomputed_scores(args, table, spec)
            except Exception as exc:
                failure = record_failure_from_exception(table.name, spec.model, "precomputed", exc)
                failures.append(failure)
                print_failure(failure)
                if args.fail_fast:
                    raise
                continue
            dataset_records.append(records)
            summary_records.append(summarize_precomputed_scores(records, spec))

        write_dataset_score_tables(args.output_dir, table.name, dataset_records, summary_records)
        if summary_records:
            all_summaries.append(pd.DataFrame(summary_records))

    write_all_datasets_summary(args.output_dir, all_summaries)
    write_failures(args.output_dir, failures)


def load_or_compute_model_scores(args: argparse.Namespace, table: DatasetTable, spec):
    """Reuse existing per-model scores when possible, otherwise compute and save them"""
    clean_path = model_scores_path(args.output_dir, table.name, spec)
    if args.skip_existing and clean_path.exists():
        print(f"[{table.name}/{spec.key}] loading existing scores from {clean_path}")
        return load_model_scores(clean_path, table.name, spec)

    if args.ensure_nltk_data and spec.backend == "gensim":
        ensure_nltk_data()

    print(f"[{table.name}/{spec.key}] loading model: {spec.display_name}")
    encoder = EmbeddingEncoder(
        spec,
        device=args.device,
        max_length=args.max_length,
    )
    # Batching is only for throughput, score_dataset_table keeps rows aligned
    records = score_dataset_table(table, spec, encoder, batch_size=args.batch_size)
    records = drop_invalid_score_rows(
        records,
        keep_invalid=args.keep_invalid,
        allow_partial_scores=table.allow_partial_answers,
    )
    write_model_scores(records, clean_path)
    del encoder
    return records


def load_and_write_precomputed_scores(args: argparse.Namespace, table: DatasetTable, spec):
    """Import an external score file and save it beside computed model scores"""
    print(f"[{table.name}/{spec.model}] importing precomputed scores: {spec.path}")
    records = load_precomputed_scores(table, spec, denormalize=args.denormalize_precomputed_scores)
    if spec.model != "snpmi":
        records = drop_invalid_score_rows(records, keep_invalid=args.keep_invalid)
    path = model_scores_dir(args.output_dir, table.name) / f"{safe_filename(spec.model)}.csv"
    write_model_scores(records, path)
    return records


def validate_precomputed_datasets(dataset_specs, precomputed_specs) -> None:
    """Ensure every external score file is attached to a requested dataset"""
    requested = {name for name, _ in dataset_specs}
    unknown = sorted(set(precomputed_specs) - requested)
    if unknown:
        raise ValueError(f"Precomputed scores were provided for unknown datasets: {unknown}")


def apply_transformer_pooling(model_specs, pooling_type: str):
    """Apply --pooling-type to manually pooled transformer models"""
    # SentenceTransformer models already carry their own pooling inside the library
    return [
        replace(spec, pooling=pooling_type) if spec.backend == "transformer" else spec
        for spec in model_specs
    ]


def ensure_nltk_data() -> None:
    """Download NLTK tokenizer resources only when they are missing."""
    try:
        import nltk
    except ImportError as exc:
        raise RuntimeError("Install nltk before using --ensure-nltk-data") from exc

    for package, resource in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            print(f"Downloading missing NLTK data: {package}")
            if not nltk.download(package):
                raise RuntimeError(f"Failed to download NLTK data package: {package}")
            nltk.data.find(resource)


def warn_missing_hf_token(model_specs) -> None:
    """Warn when requested models are known to require Hugging Face auth."""
    requested_gated = sorted(spec.key for spec in model_specs if spec.key in GATED_MODEL_KEYS)
    if requested_gated and not os.getenv("HF_TOKEN"):
        gated = ", ".join(requested_gated)
        print(
            f"Warning: HF_TOKEN is missing, gated model(s) will be skipped: {gated}",
            file=sys.stderr,
        )


def missing_gated_model_token(spec) -> bool:
    """Return whether a model should be skipped because auth is unavailable."""
    return spec.key in GATED_MODEL_KEYS and not os.getenv("HF_TOKEN")


def record_failure_from_exception(
    dataset: str,
    model: str,
    stage: str,
    exc: Exception,
) -> dict[str, str]:
    """Convert an exception to the stable failure table schema."""
    return record_failure(dataset, model, stage, type(exc).__name__, str(exc))


def record_failure(
    dataset: str,
    model: str,
    stage: str,
    error_type: str,
    error: str,
) -> dict[str, str]:
    """Create one model/import failure record."""
    return {
        "dataset": dataset,
        "model": model,
        "stage": stage,
        "error_type": error_type,
        "error": error,
    }


def print_failure(failure: dict[str, str]) -> None:
    """Print a compact failure line while the run continues."""
    print(
        f"[{failure['dataset']}/{failure['model']}] {failure['stage']} failed: "
        f"{failure['error_type']}: {failure['error']}",
        file=sys.stderr,
    )


def write_failures(output_dir: Path, failures: list[dict[str, str]]) -> None:
    """Write a CSV of skipped/failed model runs for post-run inspection."""
    if not failures:
        return

    path = output_dir / FAILURES_FILENAME
    pd.DataFrame(failures).to_csv(path, index=False)
    print(
        f"\nCompleted with {len(failures)} model/import failure(s). "
        f"Saved failure report: {path}"
    )


def print_dataset_header(name: str, row_count: int) -> None:
    """Print a compact progress header for one dataset"""
    print(f"\n{'=' * 70}")
    print(f"Dataset: {name} (triplet, {row_count} rows)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
