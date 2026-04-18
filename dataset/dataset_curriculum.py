"""
dataset_curriculum.py — Curriculum learning dataset for MATCHA.

Organizes training datasets by difficulty level and progressively exposes
harder examples as training progresses. Each difficulty level contains one
or more parquet-backed datasets; samples are drawn from the current level
using weighted random sampling.

Curriculum state (level, progress, sample counts) can be saved to and
restored from checkpoints for seamless training resumption.
"""

import os
import random
import threading

import torch
import pyarrow.parquet as pq
from torch.utils.data import Dataset

# Thread-safe lock for printing dataset statistics from DataLoader workers
_print_lock = threading.Lock()


class StreamingCurriculumDataset(Dataset):
    """Parquet-backed dataset with difficulty-based curriculum progression.

    Datasets are grouped into difficulty levels (from a JSON config). At any
    point, only datasets at the current level are sampled. The level advances
    based on epoch progress (time-based) or model performance.

    Args:
        data_path: Root directory containing per-dataset subdirectories,
                   each with a '{name}_train.parquet' file.
        tokenizer: HuggingFace tokenizer instance.
        dataset_config: Dict mapping dataset names to their config
                        (must include 'difficulty' and 'file' keys).
        contexual_dim: Maximum token sequence length for padding/truncation.
        max_samples: If set, cap the dataset to this many rows.
        current_level: Initial curriculum difficulty level.
    """

    def __init__(self, data_path, tokenizer, dataset_config, contexual_dim,
                 max_samples=None, current_level=1):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.contexual_dim = contexual_dim
        self.max_samples = max_samples

        # Per-dataset sampling statistics
        self.sample_counts = {}
        self.total_samples = 0
        self._last_print = 0

        # Organize datasets into difficulty levels
        self.levels = self._organize_by_difficulty(dataset_config)
        self.current_level = current_level
        self.level_progress = 0.0

        # Build a flat list of (level, dataset_info) and compute lengths
        self.all_datasets = []
        for level, datasets in self.levels.items():
            for dataset in datasets:
                dataset['length'] = self._get_dataset_length(dataset)
                self.sample_counts[dataset['name']] = 0
                self.all_datasets.append((level, dataset))

        # Compute sampling weights for the current level
        self._update_sampling_weights()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _organize_by_difficulty(self, dataset_config):
        """Group datasets by their integer difficulty level, sorted ascending."""
        levels = {}
        for name, cfg in dataset_config.items():
            difficulty = int(cfg['difficulty'])
            if difficulty not in levels:
                levels[difficulty] = []
            levels[difficulty].append({
                'name': name,
                'file_type': cfg['file'],
                'weight': 1.0,
            })
        return {k: levels[k] for k in sorted(levels.keys())}

    def _get_dataset_length(self, dataset_info):
        """Return the number of rows in a dataset's parquet file."""
        name = dataset_info['name']
        parquet_path = f"{self.data_path}/{name}/{name}_train.parquet"
        return pq.read_table(parquet_path, columns=[]).num_rows

    def _update_sampling_weights(self):
        """Recompute normalized cumulative weights for the current level."""
        filtered = [(lvl, d) for lvl, d in self.all_datasets if lvl == self.current_level]
        weights = [d['weight'] for _, d in filtered]
        total = sum(weights)
        cumulative = []
        acc = 0.0
        for w in weights:
            acc += w / total
            cumulative.append(acc)
        self.filtered_datasets = filtered
        self.cumulative_weights = cumulative

    # ------------------------------------------------------------------
    # Curriculum progression
    # ------------------------------------------------------------------

    def update_curriculum(self, progress, performance=None):
        """Advance the curriculum level based on progress or performance.

        Args:
            progress: Float in [0, 1] representing epoch fraction completed.
            performance: Optional metric; if > 0.8, advance one level.
        """
        prev_level = self.current_level

        if performance is not None:
            # Performance-based advancement
            if performance > 0.8 and self.current_level < max(self.levels.keys()):
                self.current_level += 1
                self.level_progress = 0.0
        else:
            # Time-based progression (fallback)
            total_levels = len(self.levels)
            target_level = min(int(progress * total_levels) + 1, total_levels)
            print(f"Target level: {target_level}, Current level: {self.current_level}, Progress: {progress*100:.2f}")
            if target_level > self.current_level:
                self.current_level = target_level
                self.level_progress = (progress * total_levels) % 1.0

        if prev_level != self.current_level:
            print(f"\nCurriculum ADVANCING from Level {prev_level} to Level {self.current_level}\n")
            self._update_sampling_weights()
            self.print_dataset_stats(force=True)

    def get_curriculum_state(self):
        """Return a serializable dict of the current curriculum state."""
        return {
            'current_level': self.current_level,
            'level_progress': self.level_progress,
            'sample_counts': self.sample_counts,
            'total_samples': self.total_samples,
        }

    def load_curriculum_state(self, state_dict):
        """Restore curriculum state from a checkpoint dict."""
        self.current_level = state_dict.get('current_level', 1)
        self.level_progress = state_dict.get('level_progress', 0.0)
        self.sample_counts = state_dict.get('sample_counts', {})
        self.total_samples = state_dict.get('total_samples', 0)
        self._update_sampling_weights()
        self.print_dataset_stats(force=True)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _stream_sample(self, dataset_info, index):
        """Read a single (premise, correct, incorrect) triplet from parquet.

        Locates the correct row group for the given absolute index, reads it,
        and returns the three text fields.
        """
        name = dataset_info['name']
        self.sample_counts[name] += 1
        self.total_samples += 1
        self.print_dataset_stats()

        parquet_path = f"{self.data_path}/{name}/{name}_train.parquet"
        with pq.ParquetFile(parquet_path) as pf:
            row_group_size = pf.metadata.row_group(0).num_rows
            row_group = index // row_group_size
            relative_idx = index % row_group_size
            table = pf.read_row_group(row_group, columns=['premise', 'correct_answer', 'incorrect_answer'])
            row = table.to_pandas().iloc[relative_idx]
            return row['premise'], row['correct_answer'], row['incorrect_answer']

    def _encode(self, text):
        """Tokenize a single text string into padded/truncated input IDs."""
        return self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.contexual_dim,
        ).input_ids.squeeze()

    def __getitem__(self, idx):
        """Return a tokenized triplet sampled from the current difficulty level.

        Note: The idx is used only for bounds checking; the actual sample is
        drawn randomly from the current level's datasets using weighted sampling.
        """
        if self.max_samples is not None and idx >= self.max_samples:
            raise IndexError(f"Index {idx} is out of bounds for max_samples {self.max_samples}")

        # Weighted random selection of a dataset within the current level
        rand_val = random.random()
        selected_idx = next(i for i, cw in enumerate(self.cumulative_weights) if rand_val <= cw)
        level, dataset_info = self.filtered_datasets[selected_idx]

        # Random sample within the selected dataset
        rand_idx = random.randint(0, dataset_info['length'] - 1)
        premise, correct, incorrect = self._stream_sample(dataset_info, rand_idx)

        return {
            'premise': self._encode(premise),
            'correct': self._encode(correct),
            'incorrect': self._encode(incorrect),
        }

    def __len__(self):
        total_samples = sum(d['length'] for lvl, d in self.all_datasets if lvl == self.current_level)
        print(f"Total samples available at current level {self.current_level}: {total_samples}")
        if self.max_samples is not None:
            return min(total_samples, self.max_samples)
        return total_samples

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_dataset_stats(self, force=False):
        """Print sampling statistics (throttled to every 1000 samples)."""
        with _print_lock:
            if not force and self.total_samples - self._last_print < 1000:
                return
            self._last_print = self.total_samples
            print("\n=== Current Dataset Statistics ===")
            print(f"Current Level: {self.current_level} (Progress: {self.level_progress*100:.2f}%)")
            print("Active Datasets:")
            total = max(1, sum(self.sample_counts.values()))
            max_name_len = max(len(name) for name in self.sample_counts)
            for name, count in sorted(self.sample_counts.items()):
                if count == 0:
                    continue
                difficulty = next(
                    (level for level, datasets in self.levels.items()
                     if any(d['name'] == name for d in datasets)), None)
                pct = count / total * 100
                print(f"- {name:<{max_name_len}} (Level {difficulty}): {pct:5.2f}% ({count} samples)")
            print(f"Total samples processed: {self.total_samples}")
            print("===============================\n")

    def __del__(self):
        """Print final stats on cleanup."""
        try:
            self.print_dataset_stats(force=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Stack a list of sample dicts into a batched dict."""
    return {
        'premise': torch.stack([x['premise'] for x in batch]),
        'correct': torch.stack([x['correct'] for x in batch]),
        'incorrect': torch.stack([x['incorrect'] for x in batch]),
    }
