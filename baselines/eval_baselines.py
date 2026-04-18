"""
eval_baselines.py — Compare baseline metrics with MATCHA on evaluation datasets.

Computes all baseline metrics from scratch (R1-F1, R2-F1, RL-F1, METEOR,
EmbSim, BertScore, BLEURT, SimCSE, Mauve), loads MATCHA results from
eval_matcha.py output, rescales EmbSim/BLEURT/SimCSE/MATCHA to [0,1] via
(score+1)/2, and computes statistical comparison metrics (macro-F1, balanced
accuracy, Wasserstein). Mauve is corpus-level and excluded from f1/wasser/bacc.

Outputs (given --output-dir <out>):
  Per dataset (<out>/{dataset}/):
    score_df.csv                              — mean scores (metric, correct/incorrect rounded to 2dp, difference)
    {dataset}.csv                             — per-sample rescaled results (all baseline + MATCHA scores)

  Cross-dataset summaries (<out>/):
    f1_results_afterconvert_macro.csv         — macro-F1 per metric per dataset 
    wasser_results_afterconvert.csv           — Wasserstein distance per metric per dataset 
    bacc_results_afterconvert.csv             — balanced accuracy per metric per dataset 
    final_score_df.csv                        — combined (correct, incorrect) pairs and differences across all datasets

Usage:
    python eval_baselines.py --matcha-path outputs/.../eval_results --datasets snli climate_fever
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score_fn
import evaluate as hf_evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from SimCSE.simcse import SimCSE
import mauve
from scipy.stats import wasserstein_distance
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
TRIPLET_DATASETS = [
    'snli', 'multi_nli', 'truthfulqa',
    'coco-caption', 'newts', 'mednli'
]
PAIRWISE_DATASETS = ['climate_fever']
ALL_DATASETS = TRIPLET_DATASETS + PAIRWISE_DATASETS

BASELINE_METRICS = ['r1f1', 'r2f1', 'rLf1', 'meteor', 'embsim', 'bert', 'bleurt', 'simcse']
ALL_METRICS = BASELINE_METRICS + ['matcha']  # used for f1, wasser, bacc (no mauve — corpus-level)

DISPLAY_NAMES = {
    'r1f1': 'R1-F1', 'r2f1': 'R2-F1', 'rLf1': 'RL-F1',
    'meteor': 'METEOR', 'embsim': 'EmbSim', 'bert': 'BertScore',
    'bleurt': 'BLEURT', 'simcse': 'SimCSE', 'mauve': 'Mauve',
    'matcha': 'MATCHA',
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baselines with MATCHA")
    parser.add_argument('--matcha-path', required=True,
                        help='Path to MATCHA eval results directory (output of eval_matcha.py)')
    parser.add_argument('--output-dir', default='../data/convert/baselines',
                        help='Output directory for results')
    parser.add_argument('--datasets', nargs='*', default=None,
                        help=f'Datasets to evaluate. Default: all. Choices: {ALL_DATASETS}')
    parser.add_argument('--simcse-model', default='princeton-nlp/sup-simcse-bert-base-uncased',
                        help='SimCSE model name')
    parser.add_argument('--embedding-model', default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Sentence embedding model for EmbSim')
    parser.add_argument('--bleurt-model', default='Elron/bleurt-base-128',
                        help='BLEURT model name')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Batched metric calculators
# ---------------------------------------------------------------------------

def batch_rouge(generated_list, reference_list):
    """Compute ROUGE-1/2/L F1 for all pairs at once (single scorer instance)."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1, r2, rL = [], [], []
    for gen, ref in tqdm(zip(generated_list, reference_list), total=len(generated_list),
                         desc='ROUGE', leave=False):
        scores = scorer.score(ref, gen)
        r1.append(scores['rouge1'].fmeasure)
        r2.append(scores['rouge2'].fmeasure)
        rL.append(scores['rougeL'].fmeasure)
    return r1, r2, rL


def batch_meteor(meteor_metric, generated_list, reference_list):
    """Compute METEOR for all pairs in one call."""
    return meteor_metric.compute(predictions=generated_list, references=reference_list)['meteor']


def batch_embedding_similarity(embedding_model, generated_list, reference_list, batch_size=256):
    """Compute cosine similarity using batched encoding."""
    gen_embs = embedding_model.encode(generated_list, batch_size=batch_size, show_progress_bar=True)
    ref_embs = embedding_model.encode(reference_list, batch_size=batch_size, show_progress_bar=True)
    # Row-wise cosine similarity
    sims = np.sum(gen_embs * ref_embs, axis=1) / (
        np.linalg.norm(gen_embs, axis=1) * np.linalg.norm(ref_embs, axis=1) + 1e-8
    )
    return sims.tolist()


def batch_bertscore(generated_list, reference_list, model_type="distilbert-base-uncased"):
    """Compute BERTScore F1 for all pairs in one call."""
    P, R, F1 = bert_score_fn(generated_list, reference_list, model_type=model_type, verbose=True)
    return F1.tolist()


def batch_bleurt(bleurt_model, bleurt_tokenizer, generated_list, reference_list, batch_size=64):
    """Compute BLEURT scores in batches."""
    device = next(bleurt_model.parameters()).device
    all_scores = []
    for i in tqdm(range(0, len(generated_list), batch_size), desc='BLEURT',
                  total=(len(generated_list) + batch_size - 1) // batch_size, leave=False):
        batch_gen = generated_list[i:i + batch_size]
        batch_ref = reference_list[i:i + batch_size]
        inputs = bleurt_tokenizer(
            batch_gen, batch_ref,
            return_tensors='pt', max_length=128,
            padding='max_length', truncation=True,
        ).to(device)
        with torch.no_grad():
            scores = bleurt_model(**inputs)[0].squeeze(-1)
        all_scores.extend(scores.cpu().tolist() if scores.dim() > 0 else [scores.item()])
    return all_scores


def batch_simcse(model_simcse, generated_list, reference_list):
    """Compute SimCSE similarities for all pairs."""
    # SimCSE.similarity returns an N×M matrix; we need the diagonal
    sim_matrix = model_simcse.similarity(generated_list, reference_list)
    return [sim_matrix[i][i] for i in range(len(generated_list))]


def compute_mauve(generated_list, reference_list):
    """Compute corpus-level MAUVE score using GPT-2."""
    result = mauve.compute_mauve(
        p_text=generated_list,
        q_text=reference_list,
        featurize_model_name="gpt2",
        device_id=0,
    )
    return result.mauve


# ---------------------------------------------------------------------------
# Compute all baseline metrics for a dataframe
# ---------------------------------------------------------------------------

def compute_all_baseline_metrics(df, dataset, models):
    """Compute all baseline metrics using batched operations.

    For triplet datasets: computes metrics between reference<->correct and
    reference<->incorrect, producing correct_* and incorrect_* columns.
    For pairwise datasets: computes metrics between text1<->text2.
    """
    meteor_metric = models['meteor']
    embedding_model = models['embedding']
    bleurt_model = models['bleurt_model']
    bleurt_tokenizer = models['bleurt_tokenizer']
    model_simcse = models['simcse']

    def _compute_pair_metrics(gen_list, ref_list, desc):
        """Compute all baseline metrics for a list of (generated, reference) pairs."""
        n = len(gen_list)
        print(f"  [{desc}] Computing baselines for {n} pairs...")

        r1, r2, rL = batch_rouge(gen_list, ref_list)

        meteor_scores = [
            meteor_metric.compute(predictions=[g], references=[r])['meteor']
            for g, r in tqdm(zip(gen_list, ref_list), total=n, desc=f'{desc} METEOR', leave=False)
        ]

        sim_scores = batch_embedding_similarity(embedding_model, gen_list, ref_list)

        bert_scores = batch_bertscore(gen_list, ref_list)

        bleurt_scores = batch_bleurt(bleurt_model, bleurt_tokenizer, gen_list, ref_list)

        simcse_scores = batch_simcse(model_simcse, gen_list, ref_list)

        return {
            'r1f1': r1, 'r2f1': r2, 'rLf1': rL,
            'meteor': meteor_scores, 'embsim': sim_scores,
            'bert': bert_scores, 'bleurt': bleurt_scores, 'simcse': simcse_scores,
        }

    if dataset in PAIRWISE_DATASETS:
        t1_list = df['text1'].astype(str).tolist()
        t2_list = df['text2'].astype(str).tolist()
        metrics = _compute_pair_metrics(t2_list, t1_list, dataset)
        for m in BASELINE_METRICS:
            df[m] = metrics[m]
    else:
        for answer_type in ['correct', 'incorrect']:
            ref_list = df['reference'].astype(str).tolist()
            ans_list = df[answer_type].astype(str).tolist()
            metrics = _compute_pair_metrics(ans_list, ref_list, f'{dataset}/{answer_type}')
            for m in BASELINE_METRICS:
                df[f'{answer_type}_{m}'] = metrics[m]

    return df


# ---------------------------------------------------------------------------
# Helper: rename MATCHA columns for statistical functions
# ---------------------------------------------------------------------------

def _rename_matcha_cols(df, dataset):
    """Rename MATCHA columns so statistical functions can access them uniformly."""
    df = df.copy()
    if dataset in PAIRWISE_DATASETS:
        df = df.rename(columns={'matcha_score': 'matcha'})
    return df


# ---------------------------------------------------------------------------
# Statistical metrics (macro-F1, Wasserstein, balanced accuracy)
# ---------------------------------------------------------------------------

def compute_f1_score_metric(result_df, metrics, dataset=''):
    results = {}
    df = _rename_matcha_cols(result_df, dataset)

    for metric in metrics:
        if dataset in PAIRWISE_DATASETS:
            scores = df[metric]
            labels = df['label']
        else:
            correct_scores = df[f'correct_{metric}']
            incorrect_scores = df[f'incorrect_{metric}']
            scores = np.concatenate([correct_scores, incorrect_scores])
            labels = np.array([1] * len(correct_scores) + [0] * len(incorrect_scores))
        preds = (scores >= 0.5).astype(int)
        results[DISPLAY_NAMES[metric]] = f1_score(labels, preds, average='macro')
    return results


def compute_bacc_score(result_df, metrics, dataset=''):
    results = {}
    df = _rename_matcha_cols(result_df, dataset)

    for metric in metrics:
        if dataset in PAIRWISE_DATASETS:
            scores = df[metric]
            labels = df['label']
        else:
            correct_scores = df[f'correct_{metric}']
            incorrect_scores = df[f'incorrect_{metric}']
            scores = np.concatenate([correct_scores, incorrect_scores])
            labels = np.array([1] * len(correct_scores) + [0] * len(incorrect_scores))
        preds = (scores >= 0.5).astype(int)
        results[DISPLAY_NAMES[metric]] = balanced_accuracy_score(labels, preds)
    return results


def compute_wasserstein(result_df, metrics, dataset=''):
    results = {}
    df = _rename_matcha_cols(result_df, dataset)

    for metric in metrics:
        if dataset in PAIRWISE_DATASETS:
            correct = df[df['label'] == 1][metric]
            incorrect = df[df['label'] == 0][metric]
        else:
            correct = df[f'correct_{metric}']
            incorrect = df[f'incorrect_{metric}']
        results[DISPLAY_NAMES[metric]] = wasserstein_distance(correct, incorrect)
    return results


# ---------------------------------------------------------------------------
# Score summary table
# ---------------------------------------------------------------------------

def build_score_table(result_df, dataset, mauve_scores=None):
    """Build (metric, correct, incorrect, difference) summary table.

    Correct and incorrect are the original mean values (×100) rounded to 2 decimals.
    Difference is computed from the rounded values and then rounded to 2 decimals.
    Mauve is corpus-level and passed in separately via mauve_scores dict.
    """
    rows = []

    if dataset in PAIRWISE_DATASETS:
        correct_df = result_df[result_df['label'] == 1]
        incorrect_df = result_df[result_df['label'] == 0]

        for m in BASELINE_METRICS:
            rows.append({'metric': DISPLAY_NAMES[m],
                         'correct': correct_df[m].mean() * 100,
                         'incorrect': incorrect_df[m].mean() * 100})

        if mauve_scores:
            rows.append({'metric': DISPLAY_NAMES['mauve'],
                         'correct': mauve_scores['correct'] * 100,
                         'incorrect': mauve_scores['incorrect'] * 100})

        rows.append({'metric': DISPLAY_NAMES['matcha'],
                     'correct': correct_df['matcha_score'].mean() * 100,
                     'incorrect': incorrect_df['matcha_score'].mean() * 100})
    else:
        for m in BASELINE_METRICS:
            rows.append({'metric': DISPLAY_NAMES[m],
                         'correct': result_df[f'correct_{m}'].mean() * 100,
                         'incorrect': result_df[f'incorrect_{m}'].mean() * 100})

        if mauve_scores:
            rows.append({'metric': DISPLAY_NAMES['mauve'],
                         'correct': mauve_scores['correct'] * 100,
                         'incorrect': mauve_scores['incorrect'] * 100})

        rows.append({'metric': DISPLAY_NAMES['matcha'],
                     'correct': result_df['correct_matcha'].mean() * 100,
                     'incorrect': result_df['incorrect_matcha'].mean() * 100})

    RESCALED_DISPLAY = {DISPLAY_NAMES[m] for m in ['embsim', 'bleurt', 'simcse', 'matcha']}

    score_df = pd.DataFrame(rows)
    score_df['correct'] = score_df['correct'].round(2)
    score_df['incorrect'] = score_df['incorrect'].round(2)
    diff = score_df['correct'] - score_df['incorrect']
    score_df['Difference'] = diff.where(
        ~score_df['metric'].isin(RESCALED_DISPLAY), diff / 2
    ).round(2)
    score_df['(Correct, Incorrect)'] = (
        '(' + score_df['correct'].map('{:.2f}'.format) + ', '
        + score_df['incorrect'].map('{:.2f}'.format) + ')'
    )
    score_df = score_df[['metric', '(Correct, Incorrect)', 'Difference']]

    return score_df


# ---------------------------------------------------------------------------
# Rescale for statistical analysis
# ---------------------------------------------------------------------------

def rescale_for_analysis(result_df, dataset):
    """Rescale EmbSim, BLEURT, SimCSE, MATCHA from [-1,1] to [0,1] via (score+1)/2."""
    df = result_df.copy()
    if dataset in PAIRWISE_DATASETS:
        for col in ['matcha_score', 'bleurt', 'embsim', 'simcse']:
            if col in df.columns:
                df[col] = (df[col] + 1) / 2
    else:
        for prefix in ['correct_', 'incorrect_']:
            for m in ['bleurt', 'embsim', 'simcse', 'matcha']:
                col = f'{prefix}{m}'
                if col in df.columns:
                    df[col] = (df[col] + 1) / 2
    return df


# ---------------------------------------------------------------------------
# Data loading & merging
# ---------------------------------------------------------------------------

def load_matcha_triplet(matcha_path, dataset):
    """Load MATCHA results for a triplet dataset."""
    matcha_df = pd.read_csv(os.path.join(matcha_path, f'{dataset}_results.csv'))
    matcha_df = matcha_df.rename(columns={
        'premise': 'reference', 'correct_answer': 'correct', 'incorrect_answer': 'incorrect',
        'pos_sim': 'correct_matcha', 'neg_sim': 'incorrect_matcha',
    })
    matcha_df = matcha_df.drop_duplicates(['reference', 'correct', 'incorrect'])
    return matcha_df


def load_matcha_pairwise(matcha_path, dataset):
    """Load MATCHA results for a pairwise dataset."""
    matcha_df = pd.read_csv(os.path.join(matcha_path, f'{dataset}_results.csv'))
    matcha_df = matcha_df.rename(columns={'score': 'matcha_score'})
    return matcha_df



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    set_random_seed(42)

    datasets = args.datasets if args.datasets else ALL_DATASETS
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all baseline models
    print("Loading baseline models...")
    meteor_metric = hf_evaluate.load('meteor')
    embedding_model = SentenceTransformer(args.embedding_model)
    bleurt_tokenizer = AutoTokenizer.from_pretrained(args.bleurt_model)
    bleurt_model = AutoModelForSequenceClassification.from_pretrained(args.bleurt_model)
    bleurt_model.eval()
    model_simcse = SimCSE(args.simcse_model)
    print("All models loaded.")

    models = {
        'meteor': meteor_metric,
        'embedding': embedding_model,
        'bleurt_model': bleurt_model,
        'bleurt_tokenizer': bleurt_tokenizer,
        'simcse': model_simcse,
    }

    # Accumulators for cross-dataset summary tables
    f1_results, wasser_results, bacc_results = {}, {}, {}

    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Processing: {dataset}")
        print(f"{'='*50}")

        save_dir = os.path.join(args.output_dir, dataset)
        converted_csv = os.path.join(save_dir, f'{dataset}.csv')

        # Skip if already computed — reload saved results for summary tables
        if os.path.exists(converted_csv):
            print(f"Already computed, loading from {converted_csv}")
            result_df_rescaled = pd.read_csv(converted_csv)
            f1_results[dataset] = compute_f1_score_metric(result_df_rescaled, ALL_METRICS, dataset)
            wasser_results[dataset] = compute_wasserstein(result_df_rescaled, ALL_METRICS, dataset)
            bacc_results[dataset] = compute_bacc_score(result_df_rescaled, ALL_METRICS, dataset)
            continue

        os.makedirs(save_dir, exist_ok=True)

        # Load MATCHA results
        if dataset in PAIRWISE_DATASETS:
            matcha_df = load_matcha_pairwise(args.matcha_path, dataset)
        else:
            matcha_df = load_matcha_triplet(args.matcha_path, dataset)

        # Compute all baseline metrics from scratch
        result_df = matcha_df.copy()
        result_df = compute_all_baseline_metrics(result_df, dataset, models)

        print(f'[{dataset}] Samples: {len(result_df)}')

        # Compute corpus-level Mauve scores
        print(f"  [{dataset}] Computing Mauve...")
        mauve_scores = {}
        if dataset in PAIRWISE_DATASETS:
            correct_df = result_df[result_df['label'] == 1]
            incorrect_df = result_df[result_df['label'] == 0]
            mauve_scores['correct'] = compute_mauve(
                correct_df['text2'].astype(str).tolist(),
                correct_df['text1'].astype(str).tolist())
            mauve_scores['incorrect'] = compute_mauve(
                incorrect_df['text2'].astype(str).tolist(),
                incorrect_df['text1'].astype(str).tolist())
        else:
            ref_list = result_df['reference'].astype(str).tolist()
            mauve_scores['correct'] = compute_mauve(
                result_df['correct'].astype(str).tolist(), ref_list)
            mauve_scores['incorrect'] = compute_mauve(
                result_df['incorrect'].astype(str).tolist(), ref_list)


        # Build raw score table
        score_df = build_score_table(result_df, dataset, mauve_scores=mauve_scores)
        score_df.to_csv(os.path.join(save_dir, 'score_df.csv'), index=False)
        print(f"Saved score table to {save_dir}/score_df.csv")

        # Rescale for statistical analysis
        result_df_rescaled = rescale_for_analysis(result_df, dataset)

        # Save converted per-sample data
        result_df_rescaled.to_csv(os.path.join(save_dir, f'{dataset}.csv'), index=False)

        # Compute statistical metrics
        f1_results[dataset] = compute_f1_score_metric(result_df_rescaled, ALL_METRICS, dataset)
        wasser_results[dataset] = compute_wasserstein(result_df_rescaled, ALL_METRICS, dataset)
        bacc_results[dataset] = compute_bacc_score(result_df_rescaled, ALL_METRICS, dataset)


    # ---------------------------------------------------------------------------
    # Save cross-dataset summary tables (×100 for percentage)
    # ---------------------------------------------------------------------------
    summary_tables = {
        'f1_results_afterconvert_macro': f1_results,
        'wasser_results_afterconvert': wasser_results,
        'bacc_results_afterconvert': bacc_results,
    }

    stat_row_order = [DISPLAY_NAMES[m] for m in ALL_METRICS]
    for name, data in summary_tables.items():
        df = pd.DataFrame(data) * 100
        df.index.name = 'Metric'
        df = df.reindex(stat_row_order)
        df = df[[d for d in datasets if d in df.columns]]
        df.to_csv(os.path.join(args.output_dir, f'{name}.csv'))
        print(f"Saved {name}.csv")

    # Build final combined table (raw scores + rescaled difference)
    dataset_names_row, column_labels_row = [], []
    final_df = None
    for i, dataset in enumerate(datasets):
        data_path = os.path.join(args.output_dir, dataset)
        df_raw = pd.read_csv(os.path.join(data_path, 'score_df.csv'))
        df_raw.index = df_raw['metric']
        selected = df_raw[['(Correct, Incorrect)', 'Difference']].copy()
        dataset_names_row.extend([dataset, dataset])
        column_labels_row.extend(['(Correct, Incorrect)', 'Difference'])
        final_df = selected if final_df is None else pd.concat([final_df, selected], axis=1)

    if final_df is not None:
        final_df.columns = pd.MultiIndex.from_arrays([dataset_names_row, column_labels_row])
        final_df.to_csv(os.path.join(args.output_dir, 'final_score_df.csv'), index=True)
        print(f"Saved final_score_df.csv")

    print("\nBaseline evaluation complete.")
