"""Run LM embedding scoring and save plot-ready CSV outputs"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for dataset scoring and score import"""
    parser = argparse.ArgumentParser(description="Score LM embeddings and save CSVs for plotting")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Dataset input, can be repeated, example: --dataset snli=data/snli.csv",
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
    return parser.parse_args()


def main() -> None:
    """Run the scorer across requested datasets, models, and imported scores"""
    args = parse_args()
    if args.hf_token:
        # Keep model-loading code reading from the normal Hugging Face env var
        os.environ["HF_TOKEN"] = args.hf_token

    dataset_specs = parse_dataset_specs(args.dataset)
    model_specs = apply_transformer_pooling(resolve_model_specs(args.models), args.pooling_type)
    validate_hf_token(model_specs)
    precomputed_specs = parse_precomputed_score_specs(args.precomputed_score)
    validate_precomputed_datasets(dataset_specs, precomputed_specs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for dataset_name, dataset_path in dataset_specs:
        table = load_dataset_table(dataset_name, dataset_path, split=args.split)
        print_dataset_header(table.name, len(table.frame))

        dataset_records = []
        summary_records = []
        for spec in model_specs:
            records = load_or_compute_model_scores(args, table, spec)
            dataset_records.append(records)
            summary_records.append(summarize_model_scores(records, table.name, spec))

        for spec in precomputed_specs.get(table.name, []):
            records = load_and_write_precomputed_scores(args, table, spec)
            dataset_records.append(records)
            summary_records.append(summarize_precomputed_scores(records, spec))

        write_dataset_score_tables(args.output_dir, table.name, dataset_records, summary_records)
        if summary_records:
            all_summaries.append(pd.DataFrame(summary_records))

    write_all_datasets_summary(args.output_dir, all_summaries)


def load_or_compute_model_scores(args: argparse.Namespace, table: DatasetTable, spec):
    """Reuse existing per-model scores when possible, otherwise compute and save them"""
    clean_path = model_scores_path(args.output_dir, table.name, spec)
    if args.skip_existing and clean_path.exists():
        print(f"[{table.name}/{spec.key}] loading existing scores from {clean_path}")
        return load_model_scores(clean_path, table.name, spec)

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


def validate_hf_token(model_specs) -> None:
    """Fail early when a requested model is known to require Hugging Face auth."""
    requested_gated = sorted(spec.key for spec in model_specs if spec.key in GATED_MODEL_KEYS)
    if requested_gated and not os.getenv("HF_TOKEN"):
        gated = ", ".join(requested_gated)
        raise ValueError(
            f"HF_TOKEN is required for gated Hugging Face model(s): {gated}. "
            "Export HF_TOKEN=hf_... or pass --hf-token."
        )


def print_dataset_header(name: str, row_count: int) -> None:
    """Print a compact progress header for one dataset"""
    print(f"\n{'=' * 70}")
    print(f"Dataset: {name} (triplet, {row_count} rows)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
