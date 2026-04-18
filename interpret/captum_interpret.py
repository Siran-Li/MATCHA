"""
Interpretability evaluation for EmbSim, BERTScore, BLEURT, SimCSE, Mistral-7B, and MATCHA
using Captum's LayerIntegratedGradients on a single triplet.

For each metric, we run 4 directional comparisons:
    1. reference → correct   (label=Correct)
    2. correct   → reference (label=Correct)
    3. reference → incorrect (label=Incorrect)
    4. incorrect → reference (label=Incorrect)

Output structure (saved under --fig-path, default: ../outputs/interpret/):
    <fig-path>/
    ├── summary.json                        # All metrics' scores in one file
    ├── <metric_name>/                      # e.g. embsim/, bertscore/, bleurt/, ...
    │   ├── records/
    │   │   └── token_attr_records.json     # Scores per direction (see keys below)
    │   └── attr_figs/
    │       └── token_importance.html     # Captum HTML heatmap of token-level attributions

Scores in token_attr_records.json / summary.json:
    Keys follow the pattern <input>_<target>_<type>, e.g. "ref_corr_sim":
        - <input>:  the text being attributed (ref, corr, incorr)
        - <target>: the text used as the reference embedding (ref, corr, incorr)
        - <type>:
            _sim  = cosine similarity between the two sentence embeddings
                    (the metric's raw score for this pair)
            _attr = sum of Integrated Gradients attributions across all tokens
                    (how much each token in <input> contributes to the similarity)

If results already exist for a metric (both .json and .html present), that metric is skipped.
Delete the .json file to force re-computation.

Usage:
    python captum_interpret.py                                       # all metrics except MATCHA
    python captum_interpret.py --output-path <MATCHA_CHECKPOINT_DIR>  # all metrics including MATCHA
    python captum_interpret.py --metrics embsim bertscore             # specific metrics only

The --output-path flag is only required for evaluating MATCHA (it needs a trained checkpoint
directory containing config.yaml, model_config.json, and max_diff.pth).
"""

import numpy as np
import os
import json
import argparse
import random
import gc
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from captum.attr import LayerIntegratedGradients, visualization

# ── Triplet ──────────────────────────────────────────────────────────────────
REFERENCE = "Nothing happens if you swallow a watermelon seed."
CORRECT   = "The watermelon seeds pass through your digestive system."
INCORRECT = "You grow watermelons in your stomach."

LABEL_MAPPING = {0: "Incorrect", 1: "Correct"}
CONTEXTUAL_DIM = 128  # short sequences → keep small


# ── Utilities ────────────────────────────────────────────────────────────────

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_tokens(tokens, tokenizer):
    """Remove special tokens and clean subword prefixes for visualization."""
    special = set(tokenizer.all_special_tokens)
    # Also filter out pure padding tokens (e.g. [PAD], <pad>, <s>, </s>)
    cleaned = []
    for t in tokens:
        if t in special:
            continue
        # Skip pure padding (id=0 often maps to empty or special)
        if not t.strip():
            continue
        # Remove BERT-style subword prefix ##
        t = t.replace("##", "")
        # Remove GPT-style prefix
        t = t.replace("Ġ", "")
        # Remove SentencePiece prefix ▁
        t = t.replace("▁", "")
        if t:  # skip if nothing left after cleaning
            cleaned.append(t)
    return cleaned


def add_attributions_to_visualizer(attributions, tokens, diff, pred_ind,
                                   label, delta, vis_data_records):
    token_importance = attributions.sum(dim=-1).squeeze(0)
    token_importance = token_importance[:len(tokens)]
    token_importance = token_importance.cpu().detach().numpy()
    norm = np.linalg.norm(token_importance)
    if norm > 0:
        token_importance = token_importance / norm

    vis_data_records.append(visualization.VisualizationDataRecord(
        token_importance,
        round(diff * 100, 2),
        LABEL_MAPPING[pred_ind],
        LABEL_MAPPING[label],
        LABEL_MAPPING[pred_ind],
        round(float(attributions.sum()) * 100, 2),
        tokens,
        delta,
    ))
    return vis_data_records, round(float(attributions.sum()) * 100, 2), round(float(diff) * 100, 2)


# ═════════════════════════════════════════════════════════════════════════════
# 1. EmbSim  (sentence-transformers/all-MiniLM-L6-v2)
# ═════════════════════════════════════════════════════════════════════════════

def interpret_embsim(vis_data_records, model, text, reference, device, label=0):
    tokenizer = model.tokenizer
    inner_model = model._first_module().auto_model
    inner_model.to(device)

    with torch.no_grad():
        text_enc = tokenizer(text, return_tensors='pt', padding='max_length',
                             truncation=True, max_length=CONTEXTUAL_DIM)
        ref_enc = tokenizer(reference, return_tensors='pt', padding='max_length',
                            truncation=True, max_length=CONTEXTUAL_DIM)

        def pad_seq(seq, max_len):
            if seq.size(1) < max_len:
                pad = torch.zeros(1, max_len - seq.size(1), dtype=torch.long)
                return torch.cat([seq, pad], dim=1)
            return seq[:, :max_len]

        input_ids = pad_seq(text_enc['input_ids'], CONTEXTUAL_DIM).to(device)
        attention_mask = pad_seq(text_enc['attention_mask'], CONTEXTUAL_DIM).to(device)
        ref_ids = pad_seq(ref_enc['input_ids'], CONTEXTUAL_DIM).to(device)
        ref_mask = pad_seq(ref_enc['attention_mask'], CONTEXTUAL_DIM).to(device)

        ref_emb = model({'input_ids': ref_ids, 'attention_mask': ref_mask})['sentence_embedding']
        embedding_layer = inner_model.embeddings.word_embeddings

        def forward_func(input_ids, attention_mask, ref_emb=None):
            out = model({'input_ids': input_ids, 'attention_mask': attention_mask})['sentence_embedding']
            return F.cosine_similarity(ref_emb, out, dim=-1)

        diff = forward_func(input_ids, attention_mask, ref_emb).item()
        pred_ind = 1 if diff > 0 else 0

        lig = LayerIntegratedGradients(forward_func, embedding_layer)
        if diff <= 0:
            baselines = (ref_ids, ref_mask)
        else:
            baselines = (torch.zeros_like(input_ids).to(device),
                         torch.zeros_like(attention_mask).to(device))

        attributions_ig, delta = lig.attribute(
            inputs=(input_ids, attention_mask),
            baselines=baselines,
            additional_forward_args=(ref_emb,),
            return_convergence_delta=True, n_steps=50,
        )

        tokens = clean_tokens(tokenizer.convert_ids_to_tokens(input_ids[0]), tokenizer)

        del text_enc, ref_enc, ref_emb
        torch.cuda.empty_cache(); gc.collect()

        return add_attributions_to_visualizer(
            attributions_ig.cpu(), tokens, diff, pred_ind, label, delta.cpu(), vis_data_records)


# ═════════════════════════════════════════════════════════════════════════════
# 2. BERTScore  (distilbert-base-uncased)
# ═════════════════════════════════════════════════════════════════════════════

def interpret_bertscore(vis_data_records, scorer, bert_score_fn, text, reference, device, label=0):
    model = scorer._model
    tokenizer = scorer._tokenizer
    model.to(device)

    # Compute the actual BERTScore F1 (same as eval_baselines.py)
    _, _, F1 = bert_score_fn([text], [reference], model_type="distilbert-base-uncased", verbose=False)
    actual_score = F1[0].item()

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding='max_length',
                           truncation=True, max_length=CONTEXTUAL_DIM)
        ref_inputs = tokenizer(reference, return_tensors='pt', padding='max_length',
                               truncation=True, max_length=CONTEXTUAL_DIM)

        input_ids = inputs['input_ids'].to(device)
        ref_ids = ref_inputs['input_ids'].to(device)
        ref_emb = model(input_ids=ref_ids).last_hidden_state.mean(dim=1)
        embedding_layer = model.get_input_embeddings().to(device)

        def forward_func(input_ids, ref_emb=None):
            out = model(input_ids=input_ids).last_hidden_state.mean(dim=1)
            return F.cosine_similarity(ref_emb, out, dim=-1)

        pred_ind = 1 if actual_score > 0.5 else 0

        lig = LayerIntegratedGradients(forward_func, embedding_layer)
        # Use actual BERTScore F1 to decide baseline
        baseline = ref_ids if actual_score <= 0.5 else torch.zeros_like(input_ids).to(device)
        attributions_ig, delta = lig.attribute(
            inputs=input_ids, baselines=baseline,
            additional_forward_args=ref_emb,
            return_convergence_delta=True, n_steps=50,
        )

        tokens = clean_tokens(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]), tokenizer)

        del inputs, ref_inputs, ref_emb
        torch.cuda.empty_cache(); gc.collect()

        return add_attributions_to_visualizer(
            attributions_ig.cpu(), tokens, actual_score, pred_ind, label, delta.cpu(), vis_data_records)


# ═════════════════════════════════════════════════════════════════════════════
# 3. BLEURT  (Elron/bleurt-base-512)
# ═════════════════════════════════════════════════════════════════════════════

def interpret_bleurt(vis_data_records, model, tokenizer, text, reference, device, label=0):
    with torch.no_grad():
        model.to(device)
        candidate_enc = tokenizer([text], return_tensors='pt').to(device)
        reference_enc = tokenizer([reference], return_tensors='pt').to(device)

        def forward_func(input_ids, attention_mask, reference_inputs=None):
            ref_ids = reference_inputs['input_ids']
            ref_mask = reference_inputs['attention_mask']
            if ref_ids.shape[0] != input_ids.shape[0]:
                ref_ids = ref_ids.expand(input_ids.shape[0], -1)
                ref_mask = ref_mask.expand(input_ids.shape[0], -1)

            max_len = max(input_ids.shape[1], ref_ids.shape[1])
            def _pad(t, tgt):
                if t.shape[1] < tgt:
                    return torch.cat([t, torch.zeros(t.shape[0], tgt - t.shape[1],
                                                     dtype=torch.long, device=t.device)], dim=1)
                return t[:, :tgt]

            input_ids = _pad(input_ids, max_len)
            attention_mask = _pad(attention_mask, max_len)
            ref_ids = _pad(ref_ids, max_len)
            ref_mask = _pad(ref_mask, max_len)

            concat_ids = torch.cat((input_ids, ref_ids[:, 1:]), dim=1)
            concat_mask = torch.cat((attention_mask, ref_mask[:, 1:]), dim=1)
            token_type_ids = torch.cat(
                (torch.zeros_like(input_ids), torch.ones_like(ref_ids[:, 1:])), dim=1)
            outputs = model(input_ids=concat_ids, attention_mask=concat_mask,
                            token_type_ids=token_type_ids)
            return outputs.logits.view(-1)

        ref_inputs = {
            'input_ids': reference_enc['input_ids'],
            'attention_mask': reference_enc['attention_mask'],
        }
        pred_score = forward_func(
            candidate_enc['input_ids'], candidate_enc['attention_mask'], ref_inputs
        )[0].item()
        pred_ind = 1 if pred_score > 0 else 0

        embedding_layer = model.bert.embeddings.word_embeddings
        lig = LayerIntegratedGradients(forward_func, embedding_layer)

        if pred_score > 0:
            baseline_ids = torch.zeros_like(candidate_enc['input_ids']).to(device)
            baseline_mask = torch.zeros_like(candidate_enc['attention_mask']).to(device)
        else:
            input_len = candidate_enc['input_ids'].shape[1]
            ref_len = reference_enc['input_ids'].shape[1]
            if ref_len < input_len:
                pad_len = input_len - ref_len
                baseline_ids = torch.cat([reference_enc['input_ids'],
                    torch.zeros((1, pad_len), dtype=torch.long, device=device)], dim=1)
                baseline_mask = torch.cat([reference_enc['attention_mask'],
                    torch.zeros((1, pad_len), dtype=torch.long, device=device)], dim=1)
            else:
                baseline_ids = reference_enc['input_ids'][:, :input_len]
                baseline_mask = reference_enc['attention_mask'][:, :input_len]

        attributions, delta_val = lig.attribute(
            inputs=(candidate_enc['input_ids'], candidate_enc['attention_mask']),
            baselines=(baseline_ids, baseline_mask),
            additional_forward_args=(ref_inputs,),
            n_steps=50, return_convergence_delta=True,
        )

        tokens = clean_tokens(tokenizer.convert_ids_to_tokens(candidate_enc['input_ids'][0]), tokenizer)

        return add_attributions_to_visualizer(
            attributions.cpu(), tokens, pred_score, pred_ind, label, delta_val.cpu(), vis_data_records)


# ═════════════════════════════════════════════════════════════════════════════
# 4. SimCSE  (princeton-nlp/sup-simcse-bert-base-uncased)
# ═════════════════════════════════════════════════════════════════════════════

def interpret_simcse(vis_data_records, model_simcse, text, reference, device, label=0):
    model = model_simcse.model
    tokenizer = model_simcse.tokenizer
    model.to(device)

    # Compute actual SimCSE similarity (same as eval_baselines.py)
    sim_matrix = model_simcse.similarity([text], [reference])
    actual_score = float(sim_matrix[0][0])

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding='max_length',
                           truncation=True, max_length=CONTEXTUAL_DIM)
        ref_inputs = tokenizer(reference, return_tensors='pt', padding='max_length',
                               truncation=True, max_length=CONTEXTUAL_DIM)

        input_ids = inputs['input_ids'].to(device)
        ref_ids = ref_inputs['input_ids'].to(device)
        ref_emb = model(input_ids=ref_ids, return_dict=True).pooler_output
        ref_emb = ref_emb / ref_emb.norm(dim=1, keepdim=True)
        embedding_layer = model.get_input_embeddings().to(device)

        def forward_func(input_ids, ref_emb=None):
            out = model(input_ids=input_ids, return_dict=True).pooler_output
            out = out / out.norm(dim=1, keepdim=True)
            return F.cosine_similarity(ref_emb, out, dim=-1)

        diff = forward_func(input_ids, ref_emb).item()
        pred_ind = 1 if actual_score > 0 else 0

        lig = LayerIntegratedGradients(forward_func, embedding_layer)
        baseline = ref_ids if actual_score <= 0 else torch.zeros_like(input_ids).to(device)
        attributions_ig, delta = lig.attribute(
            inputs=input_ids, baselines=baseline,
            additional_forward_args=ref_emb,
            return_convergence_delta=True, n_steps=50,
        )

        tokens = clean_tokens(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]), tokenizer)

        del inputs, ref_inputs, ref_emb
        torch.cuda.empty_cache(); gc.collect()

        return add_attributions_to_visualizer(
            attributions_ig.cpu(), tokens, actual_score, pred_ind, label, delta.cpu(), vis_data_records)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Mistral-7B  (mistralai/Mistral-7B-v0.1)
# ═════════════════════════════════════════════════════════════════════════════

def interpret_mistral(vis_data_records, model, tokenizer, text, reference, device, label=0):
    # Detect the device of the embedding layer (may differ from `device` with device_map="auto")
    embed_device = model.model.embed_tokens.weight.device

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding='max_length',
                           truncation=True, max_length=CONTEXTUAL_DIM)
        ref_inputs = tokenizer(reference, return_tensors='pt', padding='max_length',
                               truncation=True, max_length=CONTEXTUAL_DIM)

        input_ids = inputs['input_ids'].to(embed_device)
        ref_ids = ref_inputs['input_ids'].to(embed_device)

        ref_out = model(input_ids=ref_ids, output_hidden_states=True)
        ref_emb = ref_out.hidden_states[-1].detach()

        embedding_layer = model.model.embed_tokens

        def forward_func(input_ids, ref_emb=None):
            out = model(input_ids=input_ids, output_hidden_states=True)
            out_emb = out.hidden_states[-1]
            # Move to same device for cosine similarity
            # unsqueeze(0) so Captum gets a 1-dim tensor, not a 0-dim scalar
            return F.cosine_similarity(
                ref_emb.to(out_emb.device), out_emb, dim=-1).mean().unsqueeze(0)

        diff = forward_func(input_ids, ref_emb).item()
        pred_ind = 1 if diff > 0 else 0

        lig = LayerIntegratedGradients(forward_func, embedding_layer)
        baseline = ref_ids if diff <= 0 else torch.zeros_like(input_ids).to(embed_device)
        # Use fewer steps + internal_batch_size=1 to reduce peak memory for 7B model
        attributions_ig, delta = lig.attribute(
            inputs=input_ids, baselines=baseline,
            additional_forward_args=ref_emb,
            return_convergence_delta=True,
            n_steps=20,
            internal_batch_size=3,
        )

        tokens = clean_tokens(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]), tokenizer)

        del inputs, ref_inputs, ref_emb, ref_out
        torch.cuda.empty_cache(); gc.collect()

        return add_attributions_to_visualizer(
            attributions_ig.cpu(), tokens, diff, pred_ind, label, delta.cpu(), vis_data_records)


# ═════════════════════════════════════════════════════════════════════════════
# 6. MATCHA  (our contrastive model)
# ═════════════════════════════════════════════════════════════════════════════

def interpret_matcha(vis_data_records, model, tokenizer, text, reference, device, label=0):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding='max_length',
                           truncation=True, max_length=CONTEXTUAL_DIM)
        ref_inputs = tokenizer(reference, return_tensors='pt', padding='max_length',
                               truncation=True, max_length=CONTEXTUAL_DIM)

        input_ids = inputs['input_ids'].to(device)
        ref_ids = ref_inputs['input_ids'].to(device)
        ref_emb = model(ref_ids)

        embedding_layer = model.word_embeddings.to(device)

        def forward_func(input_ids, ref_emb=None):
            out_emb = model(input_ids=input_ids)
            return F.cosine_similarity(ref_emb, out_emb, dim=-1)

        diff = forward_func(input_ids, ref_emb).item()
        pred_ind = 1 if diff > 0 else 0

        lig = LayerIntegratedGradients(forward_func, embedding_layer)
        baseline = ref_ids if diff <= 0 else torch.zeros_like(input_ids).to(device)
        attributions_ig, delta = lig.attribute(
            inputs=input_ids, baselines=baseline,
            additional_forward_args=ref_emb,
            return_convergence_delta=True, n_steps=50,
        )

        tokens = clean_tokens(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]), tokenizer)

        del inputs, ref_inputs, ref_emb
        torch.cuda.empty_cache(); gc.collect()

        return add_attributions_to_visualizer(
            attributions_ig.cpu(), tokens, diff, pred_ind, label, delta.cpu(), vis_data_records)


# ═════════════════════════════════════════════════════════════════════════════
# Run all metrics on a single triplet
# ═════════════════════════════════════════════════════════════════════════════

def run_metric(metric_name, interpret_fn, interpret_args, fig_path, device):
    """Run 4 directional interpretations for one metric and save HTML + JSON."""
    print(f"\n{'='*60}")
    print(f"  {metric_name}")
    print(f"{'='*60}")

    record_path = os.path.join(fig_path, metric_name, 'records')
    attr_fig_path = os.path.join(fig_path, metric_name, 'attr_figs')

    # Skip if results already exist
    json_file = os.path.join(record_path, 'token_attr_records.json')
    html_file = os.path.join(attr_fig_path, 'token_importance.html')
    if os.path.exists(json_file) and os.path.exists(html_file):
        print(f"  Results already exist, skipping. (delete {json_file} to re-run)")
        with open(json_file, 'r') as f:
            return json.load(f)

    os.makedirs(record_path, exist_ok=True)
    os.makedirs(attr_fig_path, exist_ok=True)

    attr_scores = {}
    vis_data_records = []

    # ref → correct
    vis_data_records, attr_scores['ref_corr_attr'], attr_scores['ref_corr_sim'] = \
        interpret_fn(vis_data_records, *interpret_args, REFERENCE, CORRECT, device, label=1)
    # correct → ref
    vis_data_records, attr_scores['corr_ref_attr'], attr_scores['corr_ref_sim'] = \
        interpret_fn(vis_data_records, *interpret_args, CORRECT, REFERENCE, device, label=1)
    # ref → incorrect
    vis_data_records, attr_scores['ref_incorr_attr'], attr_scores['ref_incorr_sim'] = \
        interpret_fn(vis_data_records, *interpret_args, REFERENCE, INCORRECT, device, label=0)
    # incorrect → ref
    vis_data_records, attr_scores['incorr_ref_attr'], attr_scores['incorr_ref_sim'] = \
        interpret_fn(vis_data_records, *interpret_args, INCORRECT, REFERENCE, device, label=0)

    # Save JSON
    with open(os.path.join(record_path, 'token_attr_records.json'), 'w') as f:
        json.dump(attr_scores, f, indent=2)

    # Save HTML visualisation
    vis = visualization.visualize_text(vis_data_records)
    with open(os.path.join(attr_fig_path, 'token_importance.html'), 'w') as f:
        f.write(vis.data)

    print(f"  ref→corr   sim={attr_scores['ref_corr_sim']:.2f}  attr={attr_scores['ref_corr_attr']:.2f}")
    print(f"  corr→ref   sim={attr_scores['corr_ref_sim']:.2f}  attr={attr_scores['corr_ref_attr']:.2f}")
    print(f"  ref→incorr sim={attr_scores['ref_incorr_sim']:.2f}  attr={attr_scores['ref_incorr_attr']:.2f}")
    print(f"  incorr→ref sim={attr_scores['incorr_ref_sim']:.2f}  attr={attr_scores['incorr_ref_attr']:.2f}")

    return attr_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', default=None,
                        help='Path to a trained MATCHA checkpoint directory '
                             '(contains config.yaml, model_config.json, max_diff.pth)')
    parser.add_argument('--fig-path', default=None,
                        help='Where to save outputs (default: ../outputs/interpret)')
    parser.add_argument('--metrics', nargs='*', default=None,
                        help='Subset of metrics to run. Default: all. '
                             'Choices: embsim bertscore bleurt simcse mistral matcha')
    args = parser.parse_args()

    set_random_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    fig_path = args.fig_path or os.path.join(os.path.dirname(__file__), '..', 'outputs', 'interpret')
    os.makedirs(fig_path, exist_ok=True)

    all_metrics = ['embsim', 'bertscore', 'bleurt', 'simcse', 'mistral', 'matcha']
    metrics_to_run = args.metrics if args.metrics else all_metrics

    all_scores = {}

    # ── EmbSim ───────────────────────────────────────────────────────────
    if 'embsim' in metrics_to_run:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        st_model.to(device).eval()
        all_scores['embsim'] = run_metric(
            'embsim', interpret_embsim, (st_model,), fig_path, device)
        del st_model; torch.cuda.empty_cache(); gc.collect()

    # ── BERTScore ────────────────────────────────────────────────────────
    if 'bertscore' in metrics_to_run:
        from bert_score import BERTScorer
        from bert_score import score as bert_score_fn_lib
        scorer = BERTScorer(lang="en", model_type="distilbert-base-uncased", device=device)
        all_scores['bertscore'] = run_metric(
            'bertscore', interpret_bertscore, (scorer, bert_score_fn_lib), fig_path, device)
        del scorer; torch.cuda.empty_cache(); gc.collect()

    # ── BLEURT ───────────────────────────────────────────────────────────
    if 'bleurt' in metrics_to_run:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        bleurt_tok = AutoTokenizer.from_pretrained("Elron/bleurt-base-128")
        bleurt_model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-128")
        all_scores['bleurt'] = run_metric(
            'bleurt', interpret_bleurt, (bleurt_model, bleurt_tok), fig_path, device)
        del bleurt_model, bleurt_tok; torch.cuda.empty_cache(); gc.collect()

    # ── SimCSE ───────────────────────────────────────────────────────────
    if 'simcse' in metrics_to_run:
        # Add SimCSE to path
        simcse_path = os.path.join(os.path.dirname(__file__), '..', 'baselines', 'SimCSE')
        if simcse_path not in sys.path:
            sys.path.insert(0, simcse_path)
        from simcse import SimCSE
        simcse_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        all_scores['simcse'] = run_metric(
            'simcse', interpret_simcse, (simcse_model,), fig_path, device)
        del simcse_model; torch.cuda.empty_cache(); gc.collect()

    # ── Mistral-7B ───────────────────────────────────────────────────────
    if 'mistral' in metrics_to_run:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1",
                                                     trust_remote_code=True)
        mistral_tok.pad_token = mistral_tok.eos_token
        mistral_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "18GiB", "cpu": "24GiB"},
        )
        mistral_model.eval()
        all_scores['mistral'] = run_metric(
            'mistral', interpret_mistral, (mistral_model, mistral_tok), fig_path, device)
        del mistral_model, mistral_tok; torch.cuda.empty_cache(); gc.collect()

    # ── MATCHA ───────────────────────────────────────────────────────────
    if 'matcha' in metrics_to_run:
        if args.output_path is None:
            print("\n⚠  Skipping MATCHA: --output-path not provided.")
        else:
            import yaml
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from model import ContrastiveModel

            config = yaml.load(
                open(os.path.join(args.output_path, 'config.yaml'), 'r'),
                Loader=yaml.Loader)
            with open(os.path.join(args.output_path, 'model_config.json'), 'r') as f:
                model_config = json.load(f)
            model_config = SimpleNamespace(**model_config)

            # Load backbone + tokenizer (reuse logic from eval_matcha.py)
            from eval_matcha import load_backbone
            tokenizer, backbone = load_backbone(config)

            matcha_model = ContrastiveModel(backbone, model_config)
            ckpt = torch.load(os.path.join(args.output_path, 'max_diff.pth'),
                              map_location=device)
            matcha_model.load_state_dict(ckpt['model_state_dict'])
            matcha_model.to(device).eval()

            all_scores['matcha'] = run_metric(
                'matcha', interpret_matcha, (matcha_model, tokenizer), fig_path, device)
            del matcha_model, backbone; torch.cuda.empty_cache(); gc.collect()

    # ── Summary ──────────────────────────────────────────────────────────
    summary_path = os.path.join(fig_path, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_scores, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print("Done.")


if __name__ == '__main__':
    main()
