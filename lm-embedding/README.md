# LM Embedding Evaluation

This folder contains the code that produced semantic space analysis of embeddings

The datasets are loaded from pickle files produced by:

```bash
python dataset_process/prepare_eval_datasets.py --data_path data
```

Generated outputs are written under `lm-embedding/outputs/` and ignored
by git.

## Datasets

The default dataset set matches the paper-style plotting flow:

- `snli`
- `multi_nli`
- `truthfulqa`
- `climate_fever`
- `coco-caption`
- `newts`

## Score Embedding Models

If `--models` is not provided, the scorer runs only the default model:
`s-bert`.

`--models` is an explicit override list. Pass any subset of supported models;
the provided list replaces the default instead of appending to it.

Use `--paper-models` to run every currently wired paper embedding baseline.
`MATCHA` is added with the MATCHA commands below, and `SNPMI` stays disabled
until its computation logic is added for the prepared datasets.

Supported paper embedding model keys:

| Paper label | CLI key |
| --- | --- |
| BERT (base) | `bert-mean` |
| BGE-Large | `bge-large` |
| Bilingual-embedding-large | `bilingual-embedding-large` |
| DistilBERT-NLI | `distilbert` |
| E5-Large (Multilingual) | `multilingual-e5-large-instruct` |
| E5-Large-v2 | `e5-large-v2` |
| E5-Mistral-7B | `e5-mistral-7b-instruct` |
| GloVe | `glove-6B.300d` |
| GTE-Large | `gte-large-mean` |
| Jasper | `jasper` |
| Jina-embeddings-v3 | `jina-embeddings-v3` |
| Linq-Embed-Mistral | `linq-mistral` |
| LLaMA-2-13B | `llama-2-13b-mean` |
| LLaMA-3-8B | `llama-3-8b-mean` |
| MiniLM (S-BERT) | `s-bert` |
| Mistral-7B | `mistral-7b-mean` |
| MPNet | `mpnet` |
| SFR-Mistral | `sfr-mistral` |
| Speed-7B-Instruct | `speed-embedding-7b-instruct-mean` |
| Stella | `stella` |
| T5-Large (Sentence-T5) | `snet-t5` |
| Word2Vec | `word2vec` |

Use the CLI keys exactly as listed above.

`llama-2-13b-mean` and `llama-3-8b-mean` download Meta LLaMA checkpoints
from Hugging Face, so they require a Hugging Face account with access accepted
for those model repositories. Provide the same token either per command:

```bash
python lm-embedding/score_lm_embeddings.py \
  --dataset-path data \
  --models llama-3-8b-mean \
  --hf-token YOUR_HF_TOKEN \
  --output-dir lm-embedding/outputs/test_all_models_all_datasets
```

or once in the shell for repeated commands:

```bash
export HF_TOKEN=YOUR_HF_TOKEN
```

Default paper datasets with the default model:

```bash
python lm-embedding/score_lm_embeddings.py \
  --dataset-path data \
  --output-dir lm-embedding/outputs/test_all_models_all_datasets
```

Explicit subset of models:

```bash
python lm-embedding/score_lm_embeddings.py \
  --dataset-path data \
  --datasets snli multi_nli truthfulqa climate_fever coco-caption newts \
  --models s-bert mpnet \
  --output-dir lm-embedding/outputs/test_all_models_all_datasets
```

All currently wired paper embedding models:

```bash
python lm-embedding/score_lm_embeddings.py \
  --dataset-path data \
  --paper-models \
  --output-dir lm-embedding/outputs/test_all_models_all_datasets
```

Resume without recomputing existing model CSVs:

```bash
python lm-embedding/score_lm_embeddings.py \
  --dataset-path data \
  --models s-bert mpnet \
  --skip-existing \
  --output-dir lm-embedding/outputs/test_all_models_all_datasets
```

## Add MATCHA Scores

MATCHA scores are produced by `eval_matcha.py`, not from precomputed files in
this folder. This CLI can execute `../eval_matcha.py` and import the generated
CSV outputs into the `lm-embedding` result layout:

```bash
python lm-embedding/score_lm_embeddings.py \
  --dataset-path data \
  --output-dir lm-embedding/outputs/test_all_models_all_datasets \
  --matcha-only \
  --run-matcha \
  --matcha-output-path path/to/run_directory
```

To import an already completed `eval_matcha.py` run:

```bash
python lm-embedding/score_lm_embeddings.py \
  --dataset-path data \
  --output-dir lm-embedding/outputs/test_all_models_all_datasets \
  --matcha-only \
  --import-matcha-results path/to/run_directory/eval_results
```

Do not add generated MATCHA score CSV/PKL files under `lm-embedding/`.

## Analyze Results

Use the paper-style analysis command after scoring/importing models. By default
it analyzes and plots the paper dataset set: `snli`, `multi_nli`, `truthfulqa`,
`climate_fever`, `coco-caption`, and `newts`.

```bash
python lm-embedding/analyze_lm_embeddings.py \
  --output-dir lm-embedding/outputs/test_all_models_all_datasets
```

Outputs:

- `lm-embedding/outputs/test_all_models_all_datasets/figures/summary_table.csv`
- `lm-embedding/outputs/test_all_models_all_datasets/figures/embedding_barplots_new/*.png`

Write only the paper-style summary table without barplots:

```bash
python lm-embedding/analyze_lm_embeddings.py \
  --output-dir lm-embedding/outputs/test_all_models_all_datasets \
  --no-barplots
```

The SNPMI baseline is intentionally not wired here yet because its computation
logic is dataset-dependent and should be added instead of checked-in score
pickles.
