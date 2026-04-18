"""
Prepare training datasets for MATCHA as Parquet files.

Converts raw datasets into (premise, correct_answer, incorrect_answer) triplets
stored as Parquet files for efficient streaming during training. Datasets come in
three forms:
  - HuggingFace streaming datasets (pairs/triplets) -> streamed and written in batches
  - Pre-processed pkl files (NLI datasets) -> loaded and written directly
  - Special formats: COCO (local JSON), Flickr30k (HF with caption grouping),
    ParaNMT (local TSV)

All datasets listed in configs/mixed.json are supported here.

Usage:
    python prepare_training_datasets.py                              # process all datasets
    python prepare_training_datasets.py --datasets snli vitaminc     # process specific ones
    python prepare_training_datasets.py --batch_size 100000           # custom batch size
"""

import os
import json
import argparse
import pickle
import random

import numpy as np
import pandas as pd
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from datasets import load_dataset, Features, Value


# ============================================================================
# Dataset definitions: column mappings, HF names, splits, and types
# ============================================================================

DATASET_DETAILS = {
    # --- HuggingFace pair/triplet datasets (streamed to Parquet) ---
    'wikianswers': {
        'type': 'pairs',
        'columns': {'anchor': 'premise', 'positive': 'correct_answer'},
        'dataset_name': 'sentence-transformers/wikianswers-duplicates',
        'split': {'train': 'train'},
    },
    'quora': {
        'type': 'triplets',
        'columns': {'anchor': 'premise', 'positive': 'correct_answer', 'negative': 'incorrect_answer'},
        'dataset_name': 'sentence-transformers/quora-duplicates',
        'subset': 'triplet-all',
        'split': {'train': 'train'},
    },
    'altlex': {
        'type': 'pairs',
        'columns': {'text': 'premise', 'simplified': 'correct_answer'},
        'dataset_name': 'sentence-transformers/altlex',
        'split': {'train': 'train'},
    },
    'simple-wiki': {
        'type': 'pairs',
        'columns': {'text': 'premise', 'simplified': 'correct_answer'},
        'dataset_name': 'sentence-transformers/simple-wiki',
        'split': {'train': 'train'},
    },
    'wikihow': {
        'type': 'pairs',
        'columns': {'text': 'premise', 'summary': 'correct_answer'},
        'dataset_name': 'sentence-transformers/wikihow',
        'split': {'train': 'train'},
    },
    'sts-en': {
        'type': 'pairs',
        'columns': {'sentence1': 'premise', 'sentence2': 'correct_answer', 'similarity_score': 'score'},
        'dataset_name': 'PhilipMay/stsb_multi_mt',
        'subset': 'en',
        'split': {'train': 'train', 'validation': 'validation', 'test': 'test'},
    },

    # --- Caption datasets (special processing) ---
    'coco-captions': {
        'type': 'pairs',
        'dataset_name': 'coco-captions',
        'split': {'train': 'train', 'val': 'validation'},
    },
    'flickr30k': {
        'type': 'pairs',
        'dataset_name': 'nlphuji/flickr30k',
        'split': {'train': 'train', 'val': 'validation', 'test': 'test'},
    },

    # --- Pre-processed pkl datasets (NLI-style, prepared by prepare_eval_datasets.py) ---
    'snli': {
        'type': 'pkl',
        'dataset_name': 'snli',
        'split': {'train': 'train', 'validation': 'validation', 'test': 'test'},
    },
    'multi_nli': {
        'type': 'pkl',
        'dataset_name': 'multi_nli',
        'split': {'train': 'train', 'validation_matched': 'validation'},
    },
    'mnli_mismatched': {
        'type': 'pkl',
        'dataset_name': 'mnli_mismatched',
        'split': {'train': 'train'},
    },
    'vitaminc': {
        'type': 'pkl',
        'dataset_name': 'vitaminc',
        'split': {'train': 'train', 'validation': 'validation', 'test': 'test'},
    },
    'climate_fever': {
        'type': 'pkl',
        'dataset_name': 'climate_fever',
        'split': {'train': 'train', 'validation': 'validation'},
    },
    'newts': {
        'type': 'pkl',
        'dataset_name': 'newts',
        'split': {'train': 'train', 'validation': 'validation'},
    },

    # --- Local file dataset ---
    'paranmt': {
        'type': 'pairs',
        'columns': {'premise': 'premise', 'correct_answer': 'correct_answer'},
        'dataset_name': '../data/paranmt/para-nmt-5m-processed.txt',
        'split': {'test': 'train'},
    },
}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Parquet writing helpers
# ============================================================================

def write_batch_to_parquet(writer, batch_df, output_file):
    """Write a DataFrame batch to Parquet, initializing the writer if needed."""
    table = pa.Table.from_pandas(batch_df)
    if writer is None:
        print(batch_df.head())
        writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
    writer.write_table(table)
    return writer


def write_rows_to_parquet(data_rows, output_file, batch_size, required_columns):
    """Write a list of dicts to Parquet in batches."""
    writer = None
    total_rows = len(data_rows)
    with tqdm(total=total_rows, desc="Writing to Parquet") as pbar:
        for i in range(0, total_rows, batch_size):
            batch_df = pd.DataFrame(data_rows[i:i + batch_size])
            batch_df = batch_df[required_columns].dropna()
            writer = write_batch_to_parquet(writer, batch_df, output_file)
            pbar.update(len(batch_df))
    if writer:
        writer.close()


# ============================================================================
# Batch processing: rename columns, generate incorrect answers, normalize scores
# ============================================================================

def process_batch_optimized(batch, dataset_details, required_columns):
    """Transform a raw batch into the standard triplet format."""
    if 'sts' in dataset_details['dataset_name']:
        required_columns = ['premise', 'correct_answer', 'incorrect_answer', 'score']

    batch_df = pd.DataFrame(batch)
    batch_df = batch_df.rename(columns=dataset_details['columns'])

    # For pair datasets, generate incorrect answers by sampling from other rows
    if dataset_details['type'] == 'pairs':
        correct_answers = batch_df['correct_answer'].values
        n = len(correct_answers)
        indices = np.random.randint(0, n - 1, size=n)
        indices[indices >= np.arange(n)] += 1  # Skip self
        batch_df['incorrect_answer'] = correct_answers[indices]

    # Normalize STS scores from 0-5 to 0-1
    if 'score' in batch_df.columns:
        batch_df['score'] = batch_df['score'].astype(float) / 5.0

    batch_df.dropna(subset=required_columns, inplace=True)
    return batch_df[required_columns]


# ============================================================================
# Dataset-specific processors
# ============================================================================

def process_dataset_streaming(data_stream, output_file, total_examples, split,
                              dataset_details, batch_size, required_columns):
    """Stream a HuggingFace dataset, process in batches, and write to Parquet."""
    writer = None

    with tqdm(total=total_examples, desc=f"Processing {split}") as pbar:
        batch = []
        for example in data_stream:
            batch.append(example)

            if len(batch) >= batch_size:
                batch_df = process_batch_optimized(batch, dataset_details, required_columns)
                batch_df = batch_df.reset_index(drop=True)
                writer = write_batch_to_parquet(writer, batch_df, output_file)
                batch = []
                pbar.update(batch_size)

        # Process final partial batch
        if batch:
            batch_df = process_batch_optimized(batch, dataset_details, required_columns)
            if writer:
                writer.write_table(pa.Table.from_pandas(batch_df))
            else:
                pq.write_table(pa.Table.from_pandas(batch_df), output_file, compression='snappy')
            pbar.update(len(batch))

    if writer:
        writer.close()


def process_pkl_dataset(dataset_examples, output_file, batch_size, required_columns):
    """Write a pre-processed DataFrame (from pkl) to Parquet in batches."""
    writer = None
    total_rows = len(dataset_examples)

    with tqdm(total=total_rows, desc="Writing to Parquet") as pbar:
        for i in range(0, total_rows, batch_size):
            batch_df = dataset_examples.loc[i:i + batch_size]
            batch_df = batch_df[required_columns].dropna()
            writer = write_batch_to_parquet(writer, batch_df, output_file)
            pbar.update(len(batch_df))

    if writer:
        writer.close()


def build_caption_triplets(image_to_captions):
    """Build (premise, correct_answer, incorrect_answer) triplets from grouped captions.
    For each caption, the correct answer is another caption of the same image,
    and the incorrect answer is a caption from a different image.
    """
    data_rows = []
    all_image_ids = list(image_to_captions.keys())

    for image_id in tqdm(all_image_ids, desc="Building caption triplets"):
        captions = image_to_captions[image_id]
        for premise in captions:
            other_captions = [c for c in captions if c != premise]
            if not other_captions:
                continue
            correct_answer = np.random.choice(other_captions)

            # Sample incorrect answer from a different image
            while True:
                random_image_id = np.random.choice(all_image_ids)
                if random_image_id != image_id:
                    incorrect_answer = np.random.choice(image_to_captions[random_image_id])
                    break

            data_rows.append({
                'premise': premise,
                'correct_answer': correct_answer,
                'incorrect_answer': incorrect_answer,
            })
    return data_rows


def process_coco_from_json(json_path, output_file, batch_size, required_columns):
    """Process COCO captions from local JSON annotation files."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Group captions by image_id
    image_to_captions = {}
    for ann in data['annotations']:
        image_to_captions.setdefault(ann['image_id'], []).append(ann['caption'])

    print(f"Building triplets from {len(image_to_captions)} images...")
    data_rows = build_caption_triplets(image_to_captions)
    write_rows_to_parquet(data_rows, output_file, batch_size, required_columns)


def process_flickr30k_split(split_name, output_file, batch_size, required_columns):
    """Process Flickr30k captions for a given split."""
    dataset_info = load_dataset("nlphuji/flickr30k", split='test')
    dataset_info = pd.DataFrame(dataset_info)
    dataset_examples = dataset_info[dataset_info['split'] == split_name].reset_index(drop=True)
    total_examples = len(dataset_examples)
    print(f"Total examples in {split_name}: {total_examples}")
    del dataset_info

    # Group captions by image_id
    image_to_captions = {}
    for i in range(len(dataset_examples)):
        image_id = dataset_examples.loc[i, 'img_id']
        captions = dataset_examples.loc[i, 'caption']
        image_to_captions[image_id] = captions
    del dataset_examples

    print(f"Building triplets from {len(image_to_captions)} images...")
    data_rows = build_caption_triplets(image_to_captions)
    write_rows_to_parquet(data_rows, output_file, batch_size, required_columns)


# ============================================================================
# Main dispatcher: routes each dataset to its appropriate processor
# ============================================================================

def process_single_dataset(dataset_name, data_path, batch_size, required_columns):
    """Process one dataset and write all its splits as Parquet files."""
    details = DATASET_DETAILS[dataset_name]
    splits = details['split']
    hf_name = details['dataset_name']

    # For pkl datasets, load the pickle file once
    pkl_data = None
    if details['type'] == 'pkl':
        pkl_file = f"{data_path}/{hf_name}.pkl"
        with open(pkl_file, 'rb') as f:
            pkl_data = pickle.load(f)

    output_dir = f"{data_path}/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_label in splits.items():
        if 'sts' in hf_name:
            output_file = f"{output_dir}/{split_label}.parquet"
        else:
            output_file = f"{output_dir}/{dataset_name}_{split_label}.parquet"

        print(f"\n  [{dataset_name}] Processing split: {split_name} -> {output_file}")

        # --- HuggingFace dataset with subset ---
        if 'subset' in details:
            subset = details['subset']
            if dataset_name == 's2orc(title-citation)':
                features = Features({col: Value('string') for col in details['columns'].keys()})
                ds = load_dataset(hf_name, data_dir=subset, split=split_name,
                                  features=features, verification_mode="no_checks",
                                  download_mode="force_redownload")
                stream = load_dataset(hf_name, data_dir=subset, split=split_name,
                                      streaming=True, features=features,
                                      verification_mode="no_checks",
                                      download_mode="force_redownload")
            else:
                ds = load_dataset(hf_name, data_dir=subset, split=split_name)
                stream = load_dataset(hf_name, data_dir=subset, split=split_name, streaming=True)
            total = len(ds)
            del ds
            print(f"  Total examples: {total}")
            process_dataset_streaming(stream, output_file, total, split_label,
                                      details, batch_size, required_columns)

        # --- Pre-processed pkl datasets ---
        elif details['type'] == 'pkl':
            df = pkl_data[split_name]
            print(f"  Total examples: {len(df)}")
            process_pkl_dataset(df, output_file, batch_size, required_columns)

        # --- ParaNMT (local TSV file) ---
        elif dataset_name == 'paranmt':
            stream = load_dataset(
                "csv", data_files=hf_name, sep="\t", header=None,
                names=["premise", "correct_answer"],
                encoding="utf-8", split="train", streaming=True,
            )
            with open(hf_name, "r", encoding="utf-8") as f:
                total = sum(1 for _ in f)
            print(f"  Total examples: {total}")
            process_dataset_streaming(stream, output_file, total, split_label,
                                      details, batch_size, required_columns)

        # --- COCO captions (local JSON) ---
        elif dataset_name == 'coco-captions':
            coco_path = '/mnt/lustre/datasets/coco/annotations'
            json_path = f'{coco_path}/captions_{split_name}2017.json'
            process_coco_from_json(json_path, output_file, batch_size, required_columns)

        # --- Flickr30k (HF dataset, grouped by image) ---
        elif dataset_name == 'flickr30k':
            process_flickr30k_split(split_name, output_file, batch_size, required_columns)

        # --- Standard HuggingFace dataset (no subset) ---
        else:
            ds = load_dataset(hf_name, split=split_name)
            total = len(ds)
            del ds
            stream = load_dataset(hf_name, split=split_name, streaming=True)
            print(f"  Total examples: {total}")
            process_dataset_streaming(stream, output_file, total, split_label,
                                      details, batch_size, required_columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training datasets for MATCHA as Parquet files")
    parser.add_argument('--datasets', nargs='*', default=None,
                        help=f"Datasets to process. Default: all. Choices: {list(DATASET_DETAILS.keys())}")
    parser.add_argument('--data_path', type=str,
                        default='../data',
                        help="Output directory for Parquet files")
    parser.add_argument('--batch_size', type=int, default=100_000,
                        help="Batch size for streaming and writing (default: 100000)")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.data_path, exist_ok=True)
    set_random_seed(args.seed)

    required_columns = ['premise', 'correct_answer', 'incorrect_answer']
    datasets_to_process = args.datasets if args.datasets else list(DATASET_DETAILS.keys())

    for name in datasets_to_process:
        if name not in DATASET_DETAILS:
            print(f"Unknown dataset: {name}. Skipping.")
            continue
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")
        process_single_dataset(name, args.data_path,
                               args.batch_size, required_columns)

    print(f"\nDone. Processed {len(datasets_to_process)} dataset(s).")
