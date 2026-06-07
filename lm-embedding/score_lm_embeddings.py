#!/usr/bin/env python3
"""Score embedding models on MATCHA-prepared evaluation datasets."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from compute_scores import drop_invalid_score_rows, score_dataset_table
from load_datasets import DEFAULT_DATASETS, parse_dataset_specs
from load_models import (
    PAPER_EMBEDDING_MODEL_KEYS,
    EmbeddingEncoder,
    available_model_keys,
    resolve_model_specs,
    with_pooling,
)
from matcha_scores import import_matcha_results, run_matcha_eval
from result_io import (
    load_model_scores,
    model_scores_path,
    write_all_datasets_summary,
    write_dataset_score_tables,
    write_model_scores,
)


FAILURES_FILENAME = "failures.csv"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Score LM embedding baselines on MATCHA eval pickles."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory containing {dataset}.pkl files from prepare_eval_datasets.py",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="Dataset names or name:path.pkl specs. Default: paper plot datasets.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Embedding model keys. If omitted, only s-bert runs. "
            f"Supported: {', '.join(available_model_keys())}"
        ),
    )
    parser.add_argument(
        "--paper-models",
        action="store_true",
        help=(
            "Run all currently wired paper embedding baselines. "
            "SNPMI and MATCHA are not embedding model keys."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "test_all_models_all_datasets",
        help="Directory for generated lm-embedding score tables.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cpu.")
    parser.add_argument(
        "--pooling",
        choices=["mean", "max", "pooler"],
        default=None,
        help="Override pooling for transformer-backed models.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-model CSVs when present.",
    )
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Keep rows with missing/invalid similarity scores.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Hugging Face token for gated model downloads. "
            "Equivalent to setting HF_TOKEN for this run."
        ),
    )
    parser.add_argument(
        "--run-matcha",
        action="store_true",
        help="Run ../eval_matcha.py and import generated MATCHA scores.",
    )
    parser.add_argument(
        "--import-matcha-results",
        type=Path,
        default=None,
        help="Import existing eval_matcha.py result CSVs from this directory.",
    )
    parser.add_argument(
        "--matcha-output-path",
        type=Path,
        default=None,
        help="MATCHA training run directory passed to eval_matcha.py --output-path.",
    )
    parser.add_argument(
        "--matcha-model-name",
        default="max_diff.pth",
        help="Checkpoint filename passed to eval_matcha.py --model-name.",
    )
    parser.add_argument(
        "--matcha-tag",
        default="eval_results",
        help="Tag passed to eval_matcha.py and used as the result subdirectory.",
    )
    parser.add_argument(
        "--matcha-python",
        type=Path,
        default=None,
        help="Python executable for running eval_matcha.py. Defaults to current Python.",
    )
    parser.add_argument(
        "--matcha-only",
        action="store_true",
        help="Skip embedding baselines and only run/import MATCHA scores.",
    )
    args = parser.parse_args()
    if args.run_matcha and args.matcha_output_path is None:
        parser.error("--run-matcha requires --matcha-output-path")
    if args.matcha_only and not (args.run_matcha or args.import_matcha_results):
        parser.error("--matcha-only requires --run-matcha or --import-matcha-results")
    if args.paper_models and args.models is not None:
        parser.error("--paper-models cannot be combined with --models")
    if args.paper_models:
        args.models = list(PAPER_EMBEDDING_MODEL_KEYS)
    elif args.models is None:
        args.models = ["s-bert"]
    return args


def main() -> None:
    """Run scoring for all requested dataset/model combinations."""
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    tables = parse_dataset_specs(args.datasets, args.dataset_path)
    specs = [with_pooling(spec, args.pooling) for spec in resolve_model_specs(args.models)]
    dataset_names = [table.name for table in tables]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    scored_datasets: set[str] = set()

    if not args.matcha_only:
        for table in tables:
            print_dataset_header(table.name, table.split, table.source_path)
            dataset_summaries_changed = False

            for spec in specs:
                try:
                    path = model_scores_path(args.output_dir, table.name, spec.key)
                    if args.skip_existing and path.exists():
                        print(f"[{table.name}/{spec.key}] using existing {path}")
                        load_model_scores(path)
                        scored_datasets.add(table.name)
                        dataset_summaries_changed = True
                        continue

                    encoder = EmbeddingEncoder(
                        spec,
                        device=args.device,
                        batch_size=args.batch_size,
                        max_length=args.max_length,
                        hf_token=hf_token,
                    )
                    scores = score_dataset_table(table, encoder, spec)
                    scores = drop_invalid_score_rows(
                        scores,
                        table.name,
                        keep_invalid=args.keep_invalid,
                    )
                    write_model_scores(path, scores)
                    scored_datasets.add(table.name)
                    dataset_summaries_changed = True
                except Exception as exc:
                    failures.append(record_failure(table.name, spec.key, exc))
                    print_failure(failures[-1])

            if dataset_summaries_changed:
                write_dataset_score_tables(args.output_dir, table.name)

    matcha_results_dir = args.import_matcha_results
    if args.run_matcha:
        matcha_results_dir = run_matcha_eval(
            matcha_output_path=args.matcha_output_path,
            dataset_path=args.dataset_path,
            datasets=dataset_names,
            tag=args.matcha_tag,
            model_name=args.matcha_model_name,
            python_executable=args.matcha_python,
        )

    if matcha_results_dir is not None:
        try:
            import_matcha_results(
                results_dir=matcha_results_dir,
                output_dir=args.output_dir,
                datasets=dataset_names,
            )
            scored_datasets.update(dataset_names)
        except Exception as exc:
            failures.append(record_failure("matcha", "matcha", exc))
            print_failure(failures[-1])

    if scored_datasets:
        write_all_datasets_summary(args.output_dir)
    if failures:
        write_failures(args.output_dir, failures)


def print_dataset_header(dataset_name: str, split: str, source_path: Path) -> None:
    """Print a compact dataset banner."""
    print(f"\n{'=' * 72}")
    print(f"Dataset: {dataset_name} | split: {split} | source: {source_path}")
    print(f"{'=' * 72}")


def record_failure(dataset_name: str, model_key: str, exc: Exception) -> dict[str, str]:
    """Create a failure record for later CSV export."""
    return {
        "dataset": dataset_name,
        "model": model_key,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


def print_failure(failure: dict[str, str]) -> None:
    """Print a failure record to stderr."""
    print(
        f"[FAILED] {failure['dataset']}/{failure['model']}: "
        f"{failure['error_type']}: {failure['message']}",
        file=sys.stderr,
    )


def write_failures(output_dir: Path, failures: list[dict[str, str]]) -> None:
    """Write failures.csv."""
    import pandas as pd

    path = output_dir / FAILURES_FILENAME
    pd.DataFrame(failures).to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
