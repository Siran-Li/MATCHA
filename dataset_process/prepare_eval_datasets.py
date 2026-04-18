"""
Prepare evaluation datasets for MATCHA.

Each dataset is converted into (premise, correct_answer, incorrect_answer) triplets
and saved as a pickle file. For NLI datasets, the pivot table groups hypotheses/evidence
by premise and label. For other datasets, incorrect answers are sampled randomly.

Supported datasets:
  - NLI-style:  snli, multi_nli, vitaminc, mednli
  - Fact-check: climate_fever
  - QA-style:   truthfulqa
  - Caption:    coco-caption
  - Summary:    newts

Usage:
    python prepare_eval_datasets.py                 # process all datasets
    python prepare_eval_datasets.py --datasets snli climate_fever  # process specific ones
"""

import os
import argparse
import pickle
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_incorrect_answers(answer_list):
    """For each item, randomly sample an incorrect answer from the rest of the list."""
    incorrect = []
    n = len(answer_list)
    for i in range(n):
        choices = answer_list[:i] + answer_list[i+1:]
        incorrect.append(random.choice(choices))
    return incorrect


def process_nli_dataset(dataset_name, hf_name, splits, label_col, premise_col, hypothesis_col, label_mapping):
    """
    Process NLI-style datasets (SNLI, MultiNLI, VitaminC, MedNLI).
    Pivots on premise to get one row per premise with correct/neutral/incorrect answers.
    """
    all_data = {}
    for split in splits:
        data = load_dataset(hf_name, split=split)
        data = pd.DataFrame(data[:])
        print(f"[{dataset_name}] {split}: {len(data)} examples")

        # Map labels to role names
        data['label_name'] = data[label_col].map(label_mapping)
        data = data.dropna(subset=['label_name'])

        # Pivot: one row per premise with columns for each label role
        df = data.pivot_table(
            index=premise_col,
            columns='label_name',
            values=hypothesis_col,
            aggfunc='first'
        ).reset_index()
        df = df.rename(columns={premise_col: 'premise'})

        # Keep only rows that have both correct and incorrect answers
        available_cols = [c for c in ['premise', 'correct_answer', 'neutral', 'incorrect_answer'] if c in df.columns]
        df = df[available_cols]
        df = df.dropna(subset=['correct_answer', 'incorrect_answer'])
        df = df.drop_duplicates(subset=['premise', 'correct_answer', 'incorrect_answer'], keep='first')
        all_data[split] = df.reset_index(drop=True)

    for split, df in all_data.items():
        print(f"  [{dataset_name}] {split} after pivot: {len(df)} rows")
    return all_data


def process_snli(data_path):
    """SNLI: Stanford Natural Language Inference (Bowman et al., 2015)"""
    label_mapping = {0: 'correct_answer', 1: 'neutral', 2: 'incorrect_answer'}
    all_data = process_nli_dataset(
        'snli', 'stanfordnlp/snli',
        splits=['train', 'validation', 'test'],
        label_col='label', premise_col='premise', hypothesis_col='hypothesis',
        label_mapping=label_mapping
    )
    with open(f"{data_path}/snli.pkl", "wb") as f:
        pickle.dump(all_data, f)
    print(f"Saved snli.pkl")


def process_multi_nli(data_path):
    """MultiNLI: Multi-Genre Natural Language Inference (Williams et al., 2018)"""
    label_mapping = {0: 'correct_answer', 1: 'neutral', 2: 'incorrect_answer'}
    all_data = process_nli_dataset(
        'multi_nli', 'nyu-mll/multi_nli',
        splits=['train', 'validation_matched', 'validation_mismatched'],
        label_col='label', premise_col='premise', hypothesis_col='hypothesis',
        label_mapping=label_mapping
    )
    with open(f"{data_path}/multi_nli.pkl", "wb") as f:
        pickle.dump(all_data, f)
    print(f"Saved multi_nli.pkl")


def process_vitaminc(data_path):
    """VitaminC: fact verification dataset (Schuster et al., 2021)"""
    label_mapping = {'SUPPORTS': 'correct_answer', 'REFUTES': 'incorrect_answer', 'NOT ENOUGH INFO': 'neutral'}
    all_data = process_nli_dataset(
        'vitaminc', 'tals/vitaminc',
        splits=['train', 'validation', 'test'],
        label_col='label', premise_col='claim', hypothesis_col='evidence',
        label_mapping=label_mapping
    )
    with open(f"{data_path}/vitaminc.pkl", "wb") as f:
        pickle.dump(all_data, f)
    print(f"Saved vitaminc.pkl")


def process_climate_fever(data_path):
    """Climate-FEVER: fact verification for climate claims (Diggelmann et al., 2020)
    Uses SUPPORTS-labeled evidence as premise-claim pairs. Incorrect answers are
    randomly sampled from other claims in the dataset.
    """
    all_data = {}

    # --- Train split ---
    df = load_dataset("Jasontth/climate_fever_plus", split='train')
    df = pd.DataFrame(df[:])
    df = df.reset_index(drop=True)

    label_mapping = {'NOT_ENOUGH_INFO': -1, 'SUPPORTS': 1, 'REFUTES': 0}
    df['label'] = df['evidence_label'].map(label_mapping)
    df = df.dropna(subset=['label'])
    df = df[df['label'] == 1].reset_index(drop=True)
    print(f"[climate_fever] train (SUPPORTS only): {len(df)} examples")

    df = df.rename(columns={'evidence': 'premise', 'claim': 'correct_answer'})
    df['incorrect_answer'] = sample_incorrect_answers(df['correct_answer'].tolist())
    df = df.dropna(subset=['correct_answer', 'incorrect_answer'])
    all_data["train"] = df[['premise', 'correct_answer', 'incorrect_answer']].reset_index(drop=True)

    # --- Validation split (valid + test combined) ---
    df_valid = load_dataset("Jasontth/climate_fever_plus", split='valid')
    df_valid = pd.DataFrame(df_valid[:])
    df_test = load_dataset("Jasontth/climate_fever_plus", split='test')
    df_test = pd.DataFrame(df_test[:])
    df = pd.concat([df_valid, df_test], axis=0).reset_index(drop=True)

    df['label'] = df['evidence_label'].map(label_mapping)
    df = df.dropna(subset=['label'])
    df = df[df['label'] == 1].reset_index(drop=True)
    print(f"[climate_fever] validation (valid+test, SUPPORTS only): {len(df)} examples")

    df = df.rename(columns={'evidence': 'premise', 'claim': 'correct_answer'})
    df['incorrect_answer'] = sample_incorrect_answers(df['correct_answer'].tolist())
    df = df.dropna(subset=['correct_answer', 'incorrect_answer'])
    all_data["validation"] = df[['premise', 'correct_answer', 'incorrect_answer']].reset_index(drop=True)

    for split, d in all_data.items():
        print(f"  [climate_fever] {split}: {len(d)} rows")
    with open(f"{data_path}/climate_fever.pkl", "wb") as f:
        pickle.dump(all_data, f)
    print(f"Saved climate_fever.pkl")


def process_mednli(data_path):
    """MedNLI: natural language inference in the clinical domain"""
    label_mapping = {'entailment': 'correct_answer', 'neutral': 'neutral', 'contradiction': 'incorrect_answer'}
    all_data = process_nli_dataset(
        'mednli', 'presencesw/mednli',
        splits=['train', 'validation', 'test'],
        label_col='gold_label', premise_col='sentence1', hypothesis_col='sentence2',
        label_mapping=label_mapping
    )
    with open(f"{data_path}/mednli.pkl", "wb") as f:
        pickle.dump(all_data, f)
    print(f"Saved mednli.pkl")


def process_truthfulqa(data_path):
    """TruthfulQA: measuring truthfulness of language models (Lin et al., 2022)"""
    data = load_dataset("truthful_qa", "generation", split='validation').to_pandas()
    data = data[['question', 'best_answer', 'correct_answers', 'incorrect_answers']]
    data['correct_answers'] = data['correct_answers'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [x])
    data['incorrect_answers'] = data['incorrect_answers'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [x])

    # Filter out trivial answers
    data['correct_answers'] = data['correct_answers'].apply(
        lambda x: [ans for ans in x if ans != "I have no commentI have no comment" and len(ans) > 2]
    )
    data['incorrect_answers'] = data['incorrect_answers'].apply(
        lambda x: [ans for ans in x if ans != "I have no comment" and len(ans) > 2]
    )
    data = data[(data['correct_answers'].apply(len) > 1) & (data['incorrect_answers'].apply(len) > 1)]
    data = data.reset_index(drop=True)

    # Use last correct answer as premise, first correct as correct_answer, first incorrect as incorrect_answer
    data['premise'] = data['correct_answers'].apply(lambda x: x[-1])
    data['correct_answer'] = data['correct_answers'].apply(lambda x: x[0])
    data['incorrect_answer'] = data['incorrect_answers'].apply(lambda x: x[0])

    all_data = {'validation': data}
    print(f"[truthfulqa] validation: {len(data)} rows")
    with open(f"{data_path}/truthfulqa.pkl", "wb") as f:
        pickle.dump(all_data, f)
    print(f"Saved truthfulqa.pkl")


def process_coco_caption(data_path):
    """COCO-Caption: image captions used as semantic similarity pairs (Lin et al., 2014)
    Samples 1k images via reservoir sampling, uses captions as premise/answer pairs.
    """
    k = 1000
    streamed = load_dataset("lmms-lab/COCO-Caption", split="val", streaming=True)

    # Reservoir sampling to get k random examples
    reservoir = []
    for i, item in enumerate(streamed):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item

    data = pd.DataFrame(reservoir)
    assert isinstance(data['answer'][0], list), "Answers should be a list of lists."
    data = data.reset_index(drop=True)

    # Use all-but-last captions as premise, last caption as correct answer
    data = data[data['answer'].apply(lambda x: len(x) > 1)]
    data['premise'] = data['answer'].apply(lambda x: ' '.join(x[:-1]))
    data['correct_answer'] = data['answer'].apply(lambda x: x[-1])

    # Sample incorrect answers from other images
    data['incorrect_answer'] = sample_incorrect_answers(data['correct_answer'].tolist())
    data = data.dropna(subset=['correct_answer', 'incorrect_answer'])
    data = data[['premise', 'correct_answer', 'incorrect_answer']].reset_index(drop=True)

    all_data = {'validation': data}
    print(f"[coco-caption] validation: {len(data)} rows")
    with open(f"{data_path}/coco-caption.pkl", "wb") as f:
        pickle.dump(all_data, f)
    print(f"Saved coco-caption.pkl")


def process_newts(data_path):
    """NEWTS: news article text summarization (Bahrainian et al., 2022)
    Each article has two summaries; pairs are formed from summary <-> source sentences.
    Incorrect answers are randomly sampled from other summaries' first sentences.
    """
    split_files = {
        'train': '../data/NEWTS/NEWTS_train_2400.csv',
        'validation': '../data/NEWTS/NEWTS_test_600.csv',
    }
    all_data = {}

    for split, filepath in split_files.items():
        df = pd.read_csv(filepath)
        print(f"[newts] {split}: {len(df)} articles")

        # Split summaries into sentences, take first sentence as initial incorrect candidate
        df['summary1_first'] = df['summary1'].apply(lambda x: x.split('.')[0])
        df['summary2_first'] = df['summary2'].apply(lambda x: x.split('.')[0])

        # Create pairs: summary -> source sentences, with first sentence as placeholder incorrect
        df_part1 = df[['summary1', 'sentences1', 'summary1_first']].rename(
            columns={'summary1': 'premise', 'sentences1': 'correct_answer', 'summary1_first': 'incorrect_answer'})
        df_part2 = df[['summary2', 'sentences2', 'summary2_first']].rename(
            columns={'summary2': 'premise', 'sentences2': 'correct_answer', 'summary2_first': 'incorrect_answer'})

        df = pd.concat([df_part1, df_part2], ignore_index=True)
        df = df.dropna(subset=['premise', 'correct_answer', 'incorrect_answer'])
        df = df[['premise', 'correct_answer', 'incorrect_answer']].reset_index(drop=True)

        # Replace placeholder incorrect answers with randomly sampled ones
        df['incorrect_answer'] = sample_incorrect_answers(df['incorrect_answer'].tolist())
        df = df.dropna(subset=['correct_answer', 'incorrect_answer'])

        # Filter out very short texts
        df = df[df['premise'].apply(lambda x: len(x) > 3)]
        df = df[df['correct_answer'].apply(lambda x: len(x) > 3)]
        df = df[df['incorrect_answer'].apply(lambda x: len(x) > 3)]
        df = df[['premise', 'correct_answer', 'incorrect_answer']].reset_index(drop=True)

        all_data[split] = df
        print(f"  [{split}] after processing: {len(df)} rows")

    with open(f"{data_path}/newts.pkl", "wb") as f:
        pickle.dump(all_data, f)
    print(f"Saved newts.pkl")


# Registry of all dataset processors
DATASET_PROCESSORS = {
    'snli': process_snli,
    'multi_nli': process_multi_nli,
    'vitaminc': process_vitaminc,
    'climate_fever': process_climate_fever,
    'mednli': process_mednli,
    'truthfulqa': process_truthfulqa,
    'coco-caption': process_coco_caption,
    'newts': process_newts,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare evaluation datasets for MATCHA")
    parser.add_argument('--datasets', nargs='*', default=None,
                        help=f"Datasets to process. Default: all. Choices: {list(DATASET_PROCESSORS.keys())}")
    parser.add_argument('--data_path', type=str, default='../data',
                        help="Output directory for pkl files (default: data)")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.data_path, exist_ok=True)
    set_random_seed(args.seed)

    datasets_to_process = args.datasets if args.datasets else list(DATASET_PROCESSORS.keys())

    for name in datasets_to_process:
        if name not in DATASET_PROCESSORS:
            print(f"Unknown dataset: {name}. Skipping.")
            continue
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")
        DATASET_PROCESSORS[name](args.data_path)

    print(f"\nDone. Processed {len(datasets_to_process)} dataset(s).")
