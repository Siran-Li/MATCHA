"""
dataset_interleave.py — Parquet-backed dataset with interleaved batch loading.

Provides StreamingDataset for random-access reads from a parquet file, and
InterleavedBatchLoader for round-robin sampling across data sources using a
pre-built index file (index_by_source.json). This ensures balanced exposure
to each source within every epoch without loading the full dataset into RAM.
"""

import os
import random
import json
from typing import Dict, Any, List, Optional, Tuple, Callable

import torch
import pyarrow.parquet as pq
from torch.utils.data import Dataset


class StreamingDataset(Dataset):
    """
    Random-access dataset backed by a Parquet file.

    Notes on performance:
    - __getitem__ (index-based) reads an entire row group then selects one row.
      This is correct but can be I/O-inefficient with a vanilla DataLoader.
    - get_item_by_position(row_group, relative_idx) is provided for loaders that
      already know (row_group, relative_idx) and want more control.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        contexual_dim: int = 128,
        max_samples: Optional[int] = None,
        score: bool = False,
        split: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.contexual_dim = contexual_dim
        self.max_samples = max_samples
        self.score = score

        if self.score:
            self.required_columns = ["premise", "correct_answer", "incorrect_answer", "score"]
        else:
            self.required_columns = ["premise", "correct_answer", "incorrect_answer"]

        if split is None:
            file_path = os.path.join(data_path, "mixed.parquet")
        else:
            file_path = os.path.join(data_path, f"{split}.parquet")

        self.parquet_file = pq.ParquetFile(file_path)
        self.total_rows = self.parquet_file.metadata.num_rows

        if self.max_samples is not None:
            self.total_rows = min(self.total_rows, self.max_samples)

        # Metadata helper (row groups may differ in size; do not assume fixed size).
        self.num_row_groups = self.parquet_file.num_row_groups

    def __len__(self) -> int:
        return self.total_rows

    def _encode(self, x: str) -> torch.Tensor:
        return (
            self.tokenizer(
                x,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.contexual_dim,
            )
            .input_ids.squeeze(0)
            .to(torch.long)
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Random access by absolute row index.

        This scans row groups to find where idx belongs, then reads that row group.
        """
        if idx >= self.total_rows:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_rows}")

        running_total = 0
        row_group_index = None
        relative_idx = None

        for rg in range(self.num_row_groups):
            group_size = self.parquet_file.metadata.row_group(rg).num_rows
            if idx < running_total + group_size:
                row_group_index = rg
                relative_idx = idx - running_total
                break
            running_total += group_size

        if row_group_index is None or relative_idx is None:
            raise IndexError(f"Index {idx} out of range after scanning row groups.")

        return self.get_item_by_position(row_group_index, relative_idx)

    def get_item_by_position(self, row_group: int, relative_idx: int) -> Dict[str, Any]:
        """
        Fetch a single item given (row_group, relative_idx).
        Used by InterleavedBatchLoader to avoid scanning row groups.
        """
        table = self.parquet_file.read_row_group(row_group, columns=self.required_columns)
        df = table.to_pandas()

        if relative_idx >= len(df):
            raise IndexError(
                f"Row group {row_group} has only {len(df)} rows, tried to access {relative_idx}"
            )

        row = df.iloc[relative_idx]

        if self.score:
            return {
                "premise": self._encode(row["premise"]),
                "correct": self._encode(row["correct_answer"]),
                "incorrect": self._encode(row["incorrect_answer"]),
                "score": torch.tensor(float(row["score"]), dtype=torch.float32),
            }
        else:
            return {
                "premise": self._encode(row["premise"]),
                "correct": self._encode(row["correct_answer"]),
                "incorrect": self._encode(row["incorrect_answer"]),
            }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "premise": torch.stack([x["premise"] for x in batch]),
        "correct": torch.stack([x["correct"] for x in batch]),
        "incorrect": torch.stack([x["incorrect"] for x in batch]),
    }


def collate_fn_with_score(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "premise": torch.stack([x["premise"] for x in batch]),
        "correct": torch.stack([x["correct"] for x in batch]),
        "incorrect": torch.stack([x["incorrect"] for x in batch]),
        "score": torch.stack([x["score"] for x in batch]),
    }


class InterleavedBatchLoader:
    """
    Iterator that yields already-collated batches by interleaving sources according to an
    index file mapping source -> list of (row_group, relative_idx).

    This avoids the worst-case pattern of DataLoader(shuffle=True) + __getitem__
    repeatedly reading many row groups for individual samples.

    Expected index file: <data_path>/index_by_source.json
      {
        "sourceA": [[row_group, rel_idx], [row_group, rel_idx], ...],
        "sourceB": ...
      }
    """

    def __init__(
        self,
        dataset: StreamingDataset,
        data_path: str,
        batch_size: int,
        strategy: str = "round_robin",
        shuffle_within: bool = True,
        collate: Optional[Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]]] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.strategy = strategy
        self.shuffle_within = shuffle_within

        # Auto-select collate if not provided.
        if collate is None:
            self.collate = collate_fn_with_score if getattr(dataset, "score", False) else collate_fn
        else:
            self.collate = collate

        index_path = os.path.join(data_path, "index_by_source.json")
        with open(index_path, "r") as f:
            raw_index = json.load(f)

        # Convert to list[tuple[int,int]]
        self.index_by_source: Dict[str, List[Tuple[int, int]]] = {}
        for src, idx_list in raw_index.items():
            parsed = [tuple(i) for i in idx_list]  # (row_group, relative_idx)
            if shuffle_within:
                random.shuffle(parsed)
            self.index_by_source[src] = parsed

        self.sources = list(self.index_by_source.keys())
        self.cursors = {src: 0 for src in self.sources}
        self.current_src_idx = 0

        if strategy == "random":
            random.shuffle(self.sources)

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        # Stop when all sources are exhausted
        if all(self.cursors[src] >= len(self.index_by_source[src]) for src in self.sources):
            raise StopIteration

        # Find next available source (round-robin)
        for _ in range(len(self.sources)):
            src = self.sources[self.current_src_idx]
            self.current_src_idx = (self.current_src_idx + 1) % len(self.sources)

            indices = self.index_by_source[src]
            cursor = self.cursors[src]

            if cursor >= len(indices):
                continue

            next_cursor = min(cursor + self.batch_size, len(indices))
            batch_indices = indices[cursor:next_cursor]
            self.cursors[src] = next_cursor

            batch = [
                self.dataset.get_item_by_position(row_group, rel_idx)
                for row_group, rel_idx in batch_indices
            ]
            return self.collate(batch)

        raise StopIteration

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.index_by_source.values()
        )
