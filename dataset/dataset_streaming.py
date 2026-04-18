"""
dataset_streaming.py — Parquet-backed streaming dataset for MATCHA.

Reads contrastive triplets (premise, correct_answer, incorrect_answer) from
a parquet file, scanning row groups on-the-fly so the full dataset need not
fit in memory. Also provides an InterleavedBatchLoader for round-robin
sampling across data sources.
"""

import os
import json
import random

import torch
import pyarrow.parquet as pq
from torch.utils.data import Dataset


class StreamingDataset(Dataset):
    """Map-style dataset that reads rows from a parquet file on demand.

    Each row group is read lazily when a sample within it is requested.
    Supports an optional 'score' column for weighted training.

    Args:
        data_path: Directory containing the parquet file.
        tokenizer: HuggingFace tokenizer instance.
        contexual_dim: Maximum token sequence length for padding/truncation.
        max_samples: If set, cap the dataset to this many rows.
        score: If True, also return a 'score' field per sample.
        split: If set, load '{split}.parquet'; otherwise load 'mixed.parquet'.
    """

    def __init__(self, data_path, tokenizer, contexual_dim=128, max_samples=None, score=False, split=None):
        self.tokenizer = tokenizer
        self.contexual_dim = contexual_dim
        self.max_samples = max_samples
        self.score = score

        if self.score:
            self.required_columns = ['premise', 'correct_answer', 'incorrect_answer', 'score']
        else:
            self.required_columns = ['premise', 'correct_answer', 'incorrect_answer']

        # Open the parquet file (metadata only — no data loaded yet)
        file_name = f'{split}.parquet' if split else 'mixed.parquet'
        file_path = os.path.join(data_path, file_name)
        self.parquet_file = pq.ParquetFile(file_path)
        self.total_rows = self.parquet_file.metadata.num_rows

        if self.max_samples is not None:
            self.total_rows = min(self.total_rows, self.max_samples)

        self.row_group_size = self.parquet_file.metadata.row_group(0).num_rows

    def __len__(self):
        return self.total_rows

    def _encode(self, text):
        """Tokenize a single text string into padded/truncated input IDs."""
        return self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.contexual_dim,
        ).input_ids.squeeze()

    def _row_to_sample(self, row):
        """Convert a DataFrame row into a dict of tokenized tensors."""
        sample = {
            'premise': self._encode(row['premise']),
            'correct': self._encode(row['correct_answer']),
            'incorrect': self._encode(row['incorrect_answer']),
        }
        if self.score:
            sample['score'] = torch.tensor(row['score'], dtype=torch.float32)
        return sample

    def __getitem__(self, idx):
        if idx >= self.total_rows:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_rows}")

        # Scan row groups to find which one contains idx
        running_total = 0
        for row_group_index in range(self.parquet_file.num_row_groups):
            group_size = self.parquet_file.metadata.row_group(row_group_index).num_rows
            if idx < running_total + group_size:
                relative_idx = idx - running_total
                table = self.parquet_file.read_row_group(row_group_index, columns=self.required_columns)
                row = table.to_pandas().iloc[relative_idx]
                return self._row_to_sample(row)
            running_total += group_size

        raise IndexError(f"Index {idx} out of range after scanning row groups.")

    def get_item_by_position(self, row_group, relative_idx):
        """Directly access a sample by row group index and offset within it.

        Used by InterleavedBatchLoader for source-aware sampling.
        """
        table = self.parquet_file.read_row_group(row_group, columns=self.required_columns)
        row = table.to_pandas().iloc[relative_idx]
        return self._row_to_sample(row)


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Stack a list of sample dicts into a batched dict."""
    return {
        'premise': torch.stack([x['premise'] for x in batch]),
        'correct': torch.stack([x['correct'] for x in batch]),
        'incorrect': torch.stack([x['incorrect'] for x in batch]),
    }


def collate_fn_with_score(batch):
    """Stack a list of sample dicts (with scores) into a batched dict."""
    return {
        'premise': torch.stack([x['premise'] for x in batch]),
        'correct': torch.stack([x['correct'] for x in batch]),
        'incorrect': torch.stack([x['incorrect'] for x in batch]),
        'score': torch.stack([x['score'] for x in batch]),
    }


# ---------------------------------------------------------------------------
# Interleaved batch loader
# ---------------------------------------------------------------------------

class InterleavedBatchLoader:
    """Iterator that yields batches in round-robin order across data sources.

    Reads a pre-built index file (index_by_source.json) that maps each source
    name to a list of (row_group, relative_idx) pairs. Batches are drawn from
    one source at a time, cycling through sources in round-robin (or random) order.

    Args:
        dataset: A StreamingDataset instance.
        data_path: Directory containing 'index_by_source.json'.
        batch_size: Number of samples per batch.
        strategy: 'round_robin' or 'random' source ordering.
        shuffle_within: Whether to shuffle indices within each source.
    """

    def __init__(self, dataset, data_path, batch_size, strategy='round_robin', shuffle_within=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.strategy = strategy
        self.shuffle_within = shuffle_within

        index_path = os.path.join(data_path, 'index_by_source.json')
        with open(index_path, 'r') as f:
            raw_index = json.load(f)

        # Parse string keys back to (row_group, relative_idx) tuples
        self.index_by_source = {}
        for src, idx_list in raw_index.items():
            parsed_indices = [tuple(i) for i in idx_list]
            if shuffle_within:
                random.shuffle(parsed_indices)
            self.index_by_source[src] = parsed_indices

        self.sources = list(self.index_by_source.keys())
        self.cursors = {src: 0 for src in self.sources}
        self.current_src_idx = 0

        if strategy == 'random':
            random.shuffle(self.sources)

    def __iter__(self):
        return self

    def __next__(self):
        # Stop when all sources are exhausted
        if all(self.cursors[src] >= len(self.index_by_source[src]) for src in self.sources):
            raise StopIteration

        # Try each source until we find one with remaining samples
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
            return collate_fn(batch)

        raise StopIteration

    def __len__(self):
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.index_by_source.values()
        )
