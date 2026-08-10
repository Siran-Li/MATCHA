# LM Embedding Evaluation

This folder contains the code of the experiments of Embedding Based Semantic Seperation

## Installation

Install the repository requirements first, as described in the main README.
Then install the LM-embedding-specific requirements from the repository root:

```bash
pip install -r lm-embedding/requirements.txt
```

This second step is needed because the embedding baseline pipeline has its own
scoring and plotting dependencies, including `gensim`, `sentence-transformers`,
and the plotting stack used by `analyze_lm_embeddings.py`.

## Datasets

The datasets are loaded from pickle files produced by:

```bash
cd dataset_process
python prepare_eval_datasets.py --data_path ../data
cd ..  # go back to the MATCHA directory
```

(For `newts`, the preparation script above expects the local input CSVs at
`data/NEWTS/NEWTS_train_2400.csv` and `data/NEWTS/NEWTS_test_600.csv`. Check the readme inside the folder dataset_process for how to retrieve.)

The default dataset set matches the paper-style plotting flow:

- `snli`
- `multi_nli`
- `truthfulqa`
- `climate_fever`
- `coco-caption`
- `newts`

The scorer can also load any prepared eval dataset registered in
`load_datasets.py`. The currently supported dataset names are:

- `snli`
- `multi_nli`
- `vitaminc`
- `mednli`
- `truthfulqa`
- `climate_fever`
- `coco-caption`
- `newts`

Generated outputs are written under `lm-embedding/outputs/` and ignored
by git.

## Score Embedding Models

Run all the commands below from the `MATCHA` repository root, or adjust the
paths for your working directory.

If `--models` is not provided, the scorer runs only the default model:
`s-bert`.

`--models` is an explicit override list. Pass any subset of supported models;
the provided list replaces the default instead of appending to it.

If `--datasets` is not provided, the scorer runs the default paper datasets
listed above. `--datasets` is also an explicit override list, so it can be used
to run a subset of datasets without changing the model selection.

Use `--paper-models` to run every currently wired paper embedding baseline.
This option does not include `MATCHA` or `SNPMI`: `MATCHA` is added with the
MATCHA commands below, and `SNPMI` stays disabled until its computation logic
is added for the prepared datasets.

Models in --paper-models:

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

`llama-2-13b-mean` and `llama-3-8b-mean` download Meta LLaMA checkpoints from
Hugging Face. Before running the full paper model set, use a Hugging Face
account that has accepted access for those model repositories and provide the
token:

```bash
export HF_TOKEN=YOUR_HF_TOKEN

python lm-embedding/score_lm_embeddings.py \
  --dataset-path data \
  --paper-models \
  --output-dir lm-embedding/outputs/test_all_models_all_datasets
```

The equivalent one-command form is:

```bash
python lm-embedding/score_lm_embeddings.py \
  --dataset-path data \
  --paper-models \
  --hf-token YOUR_HF_TOKEN \
  --output-dir lm-embedding/outputs/test_all_models_all_datasets
```

The full paper model set includes large models and may fail or be slow on some
hardware. If that happens, run smaller subsets with `--models` and reuse the
same `--output-dir`.

Default paper datasets with the default model (S-BERT):

```bash
python lm-embedding/score_lm_embeddings.py \
  --dataset-path data \
  --output-dir lm-embedding/outputs/test_all_models_all_datasets
```

Explicit subset of datasets and models:

```bash
python lm-embedding/score_lm_embeddings.py \
  --dataset-path data \
  --datasets snli newts \
  --models s-bert mpnet \
  --output-dir lm-embedding/outputs/test_all_models_all_datasets
```

Dataset choice and model choice are independent. Use `--datasets` to choose
which prepared datasets to score:

```bash
python lm-embedding/score_lm_embeddings.py \
  --dataset-path data \
  --datasets truthfulqa climate_fever coco-caption \
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

If one dataset/model pair fails, the scorer records that failure and continues
with the remaining models. At the end, failures are written to:

```text
lm-embedding/outputs/test_all_models_all_datasets/failures.csv
```

## Add MATCHA Scores

MATCHA scores are produced by `eval_matcha.py`. This CLI can execute `../eval_matcha.py` and import the generated
CSV outputs into the `lm-embedding` result layout:

Pass the completed MATCHA training run directory to `--matcha-output-path`.
This is the folder that contains `config.yaml`, `model_config.json`, and
`max_diff.pth`, for example:

```text
/absolute/path/to/MATCHA/outputs/matcha_gpt2/<training_dataset>/<MM-DD_HH-MM>
```

```bash
python lm-embedding/score_lm_embeddings.py \
  --dataset-path data \
  --output-dir lm-embedding/outputs/test_all_models_all_datasets \
  --matcha-only \
  --run-matcha \
  --matcha-output-path /absolute/path/to/MATCHA/outputs/matcha_gpt2/<training_dataset>/<MM-DD_HH-MM>
```

To import an already completed `eval_matcha.py` run:

```bash
python lm-embedding/score_lm_embeddings.py \
  --dataset-path data \
  --output-dir lm-embedding/outputs/test_all_models_all_datasets \
  --matcha-only \
  --import-matcha-results /absolute/path/to/MATCHA/outputs/matcha_gpt2/<training_dataset>/<MM-DD_HH-MM>/eval_results
```

## Analyze Results

Use the analysis command after scoring/importing models. By default
it analyzes and plots the paper dataset set: `snli`, `multi_nli`, `truthfulqa`,
`climate_fever`, `coco-caption`, and `newts`. It also writes the paper-style
threshold curves for `snli`, `multi_nli`, and `truthfulqa`.

The analysis expects all model scores to be cosine-similarity scores in the
`[-1, 1]` range. The wired embedding models and `eval_matcha.py` produce scores
in that range. If another model or post-processed result is added later, convert
it to this range before including it in the analysis.

```bash
python lm-embedding/analyze_lm_embeddings.py \
  --output-dir lm-embedding/outputs/test_all_models_all_datasets
```

Outputs:

- `lm-embedding/outputs/test_all_models_all_datasets/figures/summary_table.csv`
- `lm-embedding/outputs/test_all_models_all_datasets/figures/embedding_barplots/*.png`
- `lm-embedding/outputs/test_all_models_all_datasets/figures/threshold_curves/snli_multi_nli_truthfulqa_threshold_curves_norm=False_cutoff=None.png`

The SNPMI baseline is intentionally not wired here yet because its computation
logic is dataset-dependent and should be added instead of checked-in score
pickles.
