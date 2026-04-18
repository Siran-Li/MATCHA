"""
eval.py — Evaluate a MATCHA checkpoint on all supported evaluation datasets.

Loads a single checkpoint and evaluates it across the datasets prepared by
prepare_eval_datasets.py: snli, multi_nli, mednli, truthfulqa,
coco-caption, newts, climate_fever. Reports per-sample cosine similarity,
aggregate metrics (mean pos/neg similarity, std, F1 score), and saves
per-dataset CSV results.

Usage:
    python eval.py --output-path outputs --datasets snli newts climate_fever
"""

import os
import json
import argparse
import logging
import random
import pickle

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.data import DataLoader
from transformers import (
    GPT2Model, GPT2Tokenizer, GPTNeoForCausalLM,
    AutoTokenizer, AutoModelForCausalLM,
    RobertaTokenizer, RobertaModel, AutoModelForMaskedLM,
)
from sklearn.metrics import f1_score
from datasets import load_dataset
import pandas as pd

from dataset.dataset_contrastive import ConDataset
from model import ContrastiveModel


# ---------------------------------------------------------------------------
# Evaluation dataset registry
# Maps dataset name -> validation split name used in the pickle file
# ---------------------------------------------------------------------------
EVAL_DATASETS = {
    'snli':         'validation',
    'multi_nli':    'validation_matched',
    'mednli':       'validation',
    'truthfulqa':   'validation',
    'coco-caption': 'validation',
    'newts':        'validation',
}

# Pairwise evaluation datasets (cosine similarity between two texts with binary labels)
PAIRWISE_DATASETS = {'climate_fever'}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_random_seed(seed):
    """Set random seeds across all libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a MATCHA checkpoint")
    parser.add_argument('--output-path', required=True,
                        help='Path to the training run directory (contains config.yaml, model_config.json, checkpoints)')
    parser.add_argument('--model-name', default='max_diff.pth',
                        help='Checkpoint filename to evaluate (default: max_diff.pth)')
    parser.add_argument('--dataset-path', default=None,
                        help='Path to evaluation pkl files (default: from config)')
    all_datasets = list(EVAL_DATASETS.keys()) + list(PAIRWISE_DATASETS)
    parser.add_argument('--datasets', nargs='*', default=None,
                        help=f'Datasets to evaluate. Default: all. Choices: {all_datasets}')
    parser.add_argument('--tag', default='eval_results',
                        help='Subfolder name for saving results (default: eval_results)')
    return parser.parse_args()


def setup_logging(output_path):
    """Create output directory and configure file-based logging."""
    os.makedirs(output_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_path, 'eval.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console)


# ---------------------------------------------------------------------------
# Loss & metrics
# ---------------------------------------------------------------------------

def contrastive_loss(embedding_ref, embedding_pos, embedding_neg, margin=1.0):
    """Triplet margin loss using cosine similarity.

    Returns per-sample (not batch-averaged) pos/neg similarities for
    fine-grained analysis.
    """
    pos_sim = F.cosine_similarity(embedding_ref, embedding_pos, dim=-1)
    neg_sim = F.cosine_similarity(embedding_ref, embedding_neg, dim=-1)
    loss = torch.mean(F.relu(margin + neg_sim - pos_sim))
    return loss, pos_sim, neg_sim


def compute_f1(correct_scores, incorrect_scores, threshold=0.5):
    """Compute macro-F1 score by treating pos/neg similarity as a binary classification.

    Correct answers are labeled 1 (should have high similarity),
    incorrect answers are labeled 0. A fixed threshold is applied to
    produce binary predictions.
    """
    scores = np.concatenate([correct_scores, incorrect_scores])
    labels = np.array([1] * len(correct_scores) + [0] * len(incorrect_scores))
    preds = (scores >= threshold).astype(int)
    return f1_score(labels, preds, average='macro')


# ---------------------------------------------------------------------------
# Climate FEVER data loading & pairwise evaluation
# ---------------------------------------------------------------------------

def load_climate_fever():
    """Load and preprocess climate_fever dataset from HuggingFace.

    Combines valid+test splits, maps labels (SUPPORTS=1, REFUTES=0),
    drops NOT_ENOUGH_INFO, and balances classes to 150 samples each.
    """
    df = load_dataset("Jasontth/climate_fever_plus", split='valid')
    df = pd.DataFrame(df[:])

    df_text = load_dataset("Jasontth/climate_fever_plus", split='test')
    df_text = pd.DataFrame(df_text[:])
    df = pd.concat([df, df_text], axis=0).reset_index(drop=True)

    label_mapping = {'NOT_ENOUGH_INFO': -1, 'SUPPORTS': 1, 'REFUTES': 0}
    df['label'] = df['evidence_label'].map(label_mapping)
    df = df.dropna(subset=['label'])
    df = df[df['label'] != -1]

    # Balance classes: 150 samples each
    min_len = 150
    df_incorrect = df[df['label'] == 0].sample(min_len, random_state=42)
    df_correct = df[df['label'] == 1].sample(min_len, random_state=42)
    df = pd.concat([df_incorrect, df_correct], axis=0).reset_index(drop=True)

    df['text1'] = df['evidence']
    df['text2'] = df['claim']
    df = df[['text1', 'text2', 'label']].reset_index(drop=True)
    return df


def evaluate_pairwise(model, tokenizer, df, contexual_dim, device):
    """Evaluate pairwise cosine similarity between text1 and text2 columns."""
    model.eval()
    scores = []
    with torch.no_grad():
        for idx in tqdm(range(len(df))):
            input1 = tokenizer(
                df.loc[idx, 'text1'],
                return_tensors='pt', padding='max_length',
                truncation=True, max_length=contexual_dim,
            ).to(device)
            input2 = tokenizer(
                df.loc[idx, 'text2'],
                return_tensors='pt', padding='max_length',
                truncation=True, max_length=contexual_dim,
            ).to(device)

            embedding1 = model(input1['input_ids'])
            embedding2 = model(input2['input_ids'])
            score = F.cosine_similarity(embedding1, embedding2, dim=-1)
            scores.append(score.cpu().item())

    df = df.copy()
    df['score'] = scores
    return df


def evaluate_climate_fever(model, tokenizer, save_path, contexual_dim, device):
    """Run climate_fever evaluation and return metrics dict + scored dataframe."""
    df = load_climate_fever()
    logging.info(f"[climate_fever] samples={len(df)}")

    df = evaluate_pairwise(model, tokenizer, df, contexual_dim, device)

    df['label'] = df['label'].astype(int)
    df_avg = df.groupby('label')['score'].mean().reset_index()
    correct_mean = df_avg[df_avg['label'] == 1]['score'].values[0]
    incorrect_mean = df_avg[df_avg['label'] == 0]['score'].values[0]

    # F1: predict label=1 if score >= threshold
    threshold = (correct_mean + incorrect_mean) / 2
    preds = (df['score'] >= threshold).astype(int)
    f1 = f1_score(df['label'], preds, average='macro')

    metrics = {
        'dataset': 'climate_fever',
        'correct_sim_mean': float(correct_mean),
        'incorrect_sim_mean': float(incorrect_mean),
        'diff_mean': float(correct_mean - incorrect_mean),
        'f1': f1,
        'n_samples': len(df),
    }

    # Save per-sample results
    csv_path = os.path.join(save_path, 'climate_fever_results.csv')
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved per-sample results to {csv_path}")

    # Save aggregated display
    df_avg_display = pd.DataFrame({
        'correct': [f'{correct_mean * 100:.2f}'],
        'incorrect': [f'{incorrect_mean * 100:.2f}'],
    })
    df_avg_display['(Correct, Incorrect)'] = '(' + df_avg_display['correct'] + ', ' + df_avg_display['incorrect'] + ')'
    df_avg_display['Difference'] = f'{(correct_mean - incorrect_mean) * 100:.2f}'
    df_avg_display = df_avg_display[['(Correct, Incorrect)', 'Difference']]
    avg_csv_path = os.path.join(save_path, 'climate_fever_score_avg.csv')
    df_avg_display.to_csv(avg_csv_path, index=False)
    logging.info(f"Saved aggregated results to {avg_csv_path}")

    return metrics


# ---------------------------------------------------------------------------
# Evaluation loop (triplet-based datasets)
# ---------------------------------------------------------------------------

def evaluate(model, data_loader, device):
    """Run evaluation and return per-sample similarity lists.

    Returns:
        avg_loss: Mean triplet loss across batches.
        pos_list: List of per-sample positive cosine similarities.
        neg_list: List of per-sample negative cosine similarities.
    """
    model.eval()
    total_loss = 0
    pos_list, neg_list = [], []

    with torch.no_grad():
        for input_ref, input_pos, input_neg in tqdm(data_loader):
            input_ref = input_ref.to(device)
            input_pos = input_pos.to(device)
            input_neg = input_neg.to(device)

            embedding_ref = model(input_ref)
            embedding_pos = model(input_pos)
            embedding_neg = model(input_neg)

            loss, pos_sim, neg_sim = contrastive_loss(embedding_ref, embedding_pos, embedding_neg)
            total_loss += loss.item()
            pos_list.extend(pos_sim.tolist())
            neg_list.extend(neg_sim.tolist())

    avg_loss = total_loss / len(data_loader)
    return avg_loss, pos_list, neg_list


def evaluate_dataset(model, tokenizer, dataset_name, dataset_path, contexual_dim, batch_size, device):
    """Evaluate a single dataset and return metrics dict + per-sample results.

    Uses the split defined in EVAL_DATASETS for each dataset.
    """
    split = EVAL_DATASETS[dataset_name]

    # Load dataset via ConDataset (handles split routing internally)
    eval_dataset = ConDataset(dataset_path, tokenizer, dataset_name, split, contexual_dim)
    logging.info(f"[{dataset_name}] split={split}, samples={len(eval_dataset)}")

    data_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    avg_loss, pos_list, neg_list = evaluate(model, data_loader, device)

    pos_arr = np.array(pos_list)
    neg_arr = np.array(neg_list)
    f1 = compute_f1(pos_arr, neg_arr)

    metrics = {
        'dataset': dataset_name,
        'loss': avg_loss,
        'pos_sim_mean': float(np.mean(pos_arr)),
        'neg_sim_mean': float(np.mean(neg_arr)),
        'pos_sim_std': float(np.std(pos_arr)),
        'neg_sim_std': float(np.std(neg_arr)),
        'diff_mean': float(np.mean(pos_arr - neg_arr)),
        'f1': f1,
        'n_samples': len(pos_list),
    }
    return metrics, pos_list, neg_list


# ---------------------------------------------------------------------------
# Backbone loader (shared with training scripts)
# ---------------------------------------------------------------------------

def load_backbone(config):
    """Load tokenizer and backbone model based on config['model_name']."""
    model_name = config['model_name']

    if 'gpt' in model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(config['tokenizer_name'])
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if 'gpt2' in model_name:
            backbone = GPT2Model.from_pretrained(model_name)
        elif model_name == 'EleutherAI/gpt-neo-1.3B':
            backbone = GPTNeoForCausalLM.from_pretrained(model_name)

    elif 'Mistral' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'], trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        backbone = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    elif model_name == 'FacebookAI/roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(config['tokenizer_name'])
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        backbone = RobertaModel.from_pretrained(model_name)

    elif model_name == 'FacebookAI/xlm-roberta-large':
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
        tokenizer.pad_token = tokenizer.eos_token
        backbone = AutoModelForMaskedLM.from_pretrained('xlm-roberta-large')

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return tokenizer, backbone


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # Load training config and model config from the run directory
    config = yaml.load(open(os.path.join(args.output_path, 'config.yaml'), 'r'), Loader=yaml.Loader)
    with open(os.path.join(args.output_path, 'model_config.json'), 'r') as f:
        model_config = json.load(f)
    model_config = SimpleNamespace(**model_config)

    # Output setup
    save_path = os.path.join(args.output_path, args.tag)
    setup_logging(save_path)

    set_random_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    # Load tokenizer and backbone
    tokenizer, backbone = load_backbone(config)

    # Build contrastive model and load checkpoint
    contrastive_model = ContrastiveModel(backbone, model_config)
    checkpoint_path = os.path.join(args.output_path, args.model_name)
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    model_weights = torch.load(checkpoint_path, map_location=device)
    contrastive_model.load_state_dict(model_weights['model_state_dict'])
    contrastive_model.to(device).eval()

    # Determine which datasets to evaluate
    dataset_path = args.dataset_path or config.get('dataset_path', 'data')
    all_known = set(EVAL_DATASETS.keys()) | PAIRWISE_DATASETS
    datasets_to_eval = args.datasets if args.datasets else list(EVAL_DATASETS.keys()) + list(PAIRWISE_DATASETS)

    # Evaluate each dataset
    all_results = {}
    for dataset_name in datasets_to_eval:
        if dataset_name not in all_known:
            logging.warning(f"Unknown dataset: {dataset_name}, skipping.")
            continue

        logging.info(f"{'='*50}")
        logging.info(f"Evaluating: {dataset_name}")
        logging.info(f"{'='*50}")

        try:
            if dataset_name in PAIRWISE_DATASETS:
                # Pairwise evaluation (climate_fever)
                metrics = evaluate_climate_fever(
                    contrastive_model, tokenizer, save_path,
                    config['contexual_dim'], device,
                )
                logging.info(
                    f"[{dataset_name}] "
                    f"Correct: {metrics['correct_sim_mean']:.4f} | "
                    f"Incorrect: {metrics['incorrect_sim_mean']:.4f} | "
                    f"Diff: {metrics['diff_mean']:.4f} | F1: {metrics['f1']:.4f}"
                )
            else:
                # Triplet-based evaluation
                metrics, pos_list, neg_list = evaluate_dataset(
                    contrastive_model, tokenizer, dataset_name, dataset_path,
                    config['contexual_dim'], config['batch_size'], device,
                )
                logging.info(
                    f"[{dataset_name}] Loss: {metrics['loss']:.4f} | "
                    f"Pos: {metrics['pos_sim_mean']:.4f} (+/-{metrics['pos_sim_std']:.4f}) | "
                    f"Neg: {metrics['neg_sim_mean']:.4f} (+/-{metrics['neg_sim_std']:.4f}) | "
                    f"Diff: {metrics['diff_mean']:.4f} | F1: {metrics['f1']:.4f}"
                )

                # Save per-sample results to CSV
                split = EVAL_DATASETS[dataset_name]
                with open(f"{dataset_path}/{dataset_name}.pkl", 'rb') as f:
                    raw_data = pickle.load(f)
                data = raw_data[split]
                if hasattr(data, 'reset_index'):
                    data = data.reset_index(drop=True)

                data['pos_sim'] = pos_list
                data['neg_sim'] = neg_list
                data['difference'] = data['pos_sim'] - data['neg_sim']
                data = data.sort_values(by='difference', ascending=False)

                csv_path = os.path.join(save_path, f'{dataset_name}_results.csv')
                data.to_csv(csv_path, index=False)
                logging.info(f"Saved per-sample results to {csv_path}")

        except Exception as e:
            logging.error(f"Failed to evaluate {dataset_name}: {e}")
            continue

        all_results[dataset_name] = metrics

    # Save summary JSONL
    summary_path = os.path.join(save_path, 'eval_summary.jsonl')
    with open(summary_path, 'w') as f:
        for dataset_name, metrics in all_results.items():
            json.dump(metrics, f)
            f.write('\n')
    logging.info(f"Saved evaluation summary to {summary_path}")

    logging.info("Evaluation complete.")
