"""
dataset_contrastive.py — Contrastive triplet dataset for MATCHA.

Loads (premise, correct_answer, incorrect_answer) triplets from pickle
files produced by prepare_eval_datasets.py. Each sample is tokenized and
returned as three padded token-ID tensors.

Supported datasets:
  - snli, multi_nli, vitaminc, mednli  (NLI)
  - truthfulqa                          (factuality)
  - coco-caption                        (image captioning)
  - newts                               (news text summarization)
"""

import pickle

from torch.utils.data import Dataset

# Validation-only datasets (no train split in the pkl)
VALIDATION_ONLY_DATASETS = {
    'truthfulqa', 'coco-caption', 'newts', 'mednli',
}

# All datasets that use the standard (premise, correct_answer, incorrect_answer) columns
STANDARD_TRIPLET_DATASETS = VALIDATION_ONLY_DATASETS | {
    'snli', 'multi_nli', 'vitaminc',
}


class ConDataset(Dataset):
    """Contrastive triplet dataset.

    Args:
        dataset_path: Root directory containing pickled dataset files.
        tokenizer: HuggingFace tokenizer instance.
        dataset: Name of the dataset to load.
        split: Data split ('train', 'validation', etc.).
        contexual_dim: Maximum token sequence length for padding/truncation.
    """

    def __init__(self, dataset_path, tokenizer, dataset, split, contexual_dim):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.split = split
        self.contexual_dim = contexual_dim

        self.data = self._load_data()

    def _load_data(self):
        """Load and return the appropriate split as a DataFrame."""
        if self.dataset not in STANDARD_TRIPLET_DATASETS:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        with open(f'{self.dataset_path}/{self.dataset}.pkl', 'rb') as file:
            data = pickle.load(file)

        print(f'[{self.dataset}] Available splits: {list(data.keys())}')
        for split_name, split_df in data.items():
            print(f'  {split_name}: {len(split_df)} rows')

        return data[self.split].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return tokenized (premise, correct, incorrect) triplet as token IDs."""
        premise = self.data.loc[idx, 'premise']
        correct_answer = self.data.loc[idx, 'correct_answer']
        incorrect_answer = self.data.loc[idx, 'incorrect_answer']

        # Tokenize premise, correct answer, and incorrect answer
        encoded_premise = self.tokenizer(
            premise,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.contexual_dim,
        )
        encoded_correct = self.tokenizer(
            correct_answer,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.contexual_dim,
        )
        encoded_incorrect = self.tokenizer(
            incorrect_answer,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.contexual_dim,
        )

        return (
            encoded_premise.input_ids.squeeze(),
            encoded_correct.input_ids.squeeze(),
            encoded_incorrect.input_ids.squeeze(),
        )
