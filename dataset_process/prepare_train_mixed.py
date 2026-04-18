"""
Prepare a mixed training Parquet file by sampling from individual dataset Parquet files.

Reads the dataset config (configs/mixed.json), samples up to --sample_size rows from
each dataset's train split, combines them into a single mixed.parquet, and builds a
source index (index_by_source.json) for interleaved training.

Usage:
    python prepare_training_mix.py                                    # all defaults
    python prepare_training_mix.py --sample_size 50000                # 50k per dataset
    python prepare_training_mix.py --config ../configs/mixed.json     # custom config
"""

import os
import argparse
import json
import random

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def write_to_parquet(df, output_file, batch_size=100_000, columns=None):
    """Write a DataFrame to Parquet in batches."""
    writer = None
    total_rows = len(df)

    with tqdm(total=total_rows, desc="Writing Parquet") as pbar:
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i + batch_size][columns].dropna()
            table = pa.Table.from_pandas(batch_df)
            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
            writer.write_table(table)
            pbar.update(len(batch_df))

    if writer:
        writer.close()


def sample_from_parquet(pf, sample_size, dataset_name, columns):
    """Sample rows from a Parquet file using row-group-level sampling for efficiency."""
    total_rows = pf.metadata.num_rows
    sample_size = min(sample_size, total_rows)
    num_row_groups = pf.metadata.num_row_groups
    samples_per_group = max(1, sample_size // num_row_groups)

    sampled_batches = []
    for rg in tqdm(range(num_row_groups), desc=f"Sampling {dataset_name}", leave=False):
        group_rows = pf.metadata.row_group(rg).num_rows
        if group_rows == 0:
            continue

        indices = random.sample(range(group_rows), min(samples_per_group, group_rows))
        table = pf.read_row_group(rg, columns=columns)
        batch = table.take(indices).to_pandas()
        batch['source_dataset'] = dataset_name
        sampled_batches.append(batch)

    return pd.concat(sampled_batches, ignore_index=True)


def sample_and_combine(data_path, dataset_names, sample_size=None):
    """Sample from each dataset's Parquet file and combine into one DataFrame."""
    all_samples = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for name in dataset_names:
            # STS datasets use a different naming convention
            if 'sts' in name:
                parquet_path = os.path.join(data_path, name, "train.parquet")
                columns = ['premise', 'correct_answer', 'incorrect_answer', 'score']
            else:
                parquet_path = os.path.join(data_path, name, f"{name}_train.parquet")
                columns = ['premise', 'correct_answer', 'incorrect_answer']

            pf = pq.ParquetFile(parquet_path)
            set_size = min(sample_size, pf.metadata.num_rows) if sample_size else pf.metadata.num_rows
            print(f"  {name}: sampling {set_size} / {pf.metadata.num_rows} rows")

            futures.append(executor.submit(sample_from_parquet, pf, set_size, name, columns))

        for future in futures:
            all_samples.append(future.result())

    return pd.concat(all_samples, ignore_index=True)


def build_source_index(parquet_path, index_path):
    """Build a JSON index mapping source_dataset -> list of (row_group, row_index) pairs.
    Used by the dataloader for interleaved/curriculum training.
    """
    pf = pq.ParquetFile(parquet_path)
    row_group_size = pf.metadata.row_group(0).num_rows
    total_rows = pf.metadata.num_rows

    print(f"Building index: {total_rows} rows, {pf.metadata.num_row_groups} row groups")
    index_by_source = {}

    for i in tqdm(range(total_rows), desc="Indexing"):
        row_group = i // row_group_size
        rel_idx = i % row_group_size
        table = pf.read_row_group(row_group, columns=["source_dataset"])
        source = table.column("source_dataset")[rel_idx].as_py()
        index_by_source.setdefault(source, []).append((row_group, rel_idx))

    with open(index_path, 'w') as f:
        json.dump(index_by_source, f)

    for source, indices in index_by_source.items():
        print(f"  {source}: {len(indices)} rows")
    print(f"Saved index to {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample and mix training datasets into a single Parquet file")
    parser.add_argument('--config', type=str, default='../configs/mixed.json',
                        help="Path to dataset config JSON (default: ../configs/mixed.json)")
    parser.add_argument('--data_path', type=str,
                        default='../data',
                        help="Directory containing per-dataset Parquet files")
    parser.add_argument('--output_dir', type=str,
                        default='../data/mixed',
                        help="Output directory for mixed.parquet and index")
    parser.add_argument('--sample_size', type=int, default=100_000,
                        help="Max rows to sample per dataset (default: 100000)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_parquet = os.path.join(args.output_dir, "mixed.parquet")
    output_index = os.path.join(args.output_dir, "index_by_source.json")

    # Load dataset config
    with open(args.config, 'r') as f:
        dataset_config = json.load(f)
    dataset_names = list(dataset_config.keys())
    print(f"Mixing {len(dataset_names)} datasets: {dataset_names}")

    # Sample and combine
    combined_df = sample_and_combine(args.data_path, dataset_names, sample_size=args.sample_size)
    print(f"\nTotal combined rows: {len(combined_df)}")

    # Write mixed Parquet
    all_columns = ['premise', 'correct_answer', 'incorrect_answer', 'source_dataset']
    if 'score' in combined_df.columns:
        all_columns.append('score')
    write_to_parquet(combined_df, output_parquet, batch_size=50_000, columns=all_columns)

    # Build source index for interleaved training
    build_source_index(output_parquet, output_index)

    print(f"\nDone. Output: {output_parquet}")
