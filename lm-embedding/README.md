# LM Embedding Baselines

Compute and plot embedding-based semantic-separation scores for the MATCHA
triplet datasets. Self-contained inside `lm-embedding/`.

## Quick Start

```bash
# 1. install deps
pip install -r lm-embedding/requirements.txt

# 2. optionally export your Hugging Face token for gated models
export HF_TOKEN=hf_...

# 3. score embeddings
lm-embedding/scripts/compute.sh

# 4. plot figures
lm-embedding/scripts/analyze.sh
```

SNPMI and MATCHA scores are auto-included: files matching
`precomputed/snpmi/{dataset}_snpmi_scores.csv` and
`precomputed/matcha/{dataset}_matcha_scores.csv` are picked up automatically
for the datasets in `DATASETS`.
To override, set `SNPMI_DIR=/some/path` or `MATCHA_DIR=/some/path`, or set
`SKIP_AUTO_SNPMI=1` / `SKIP_AUTO_MATCHA=1` and pass `--precomputed-score` flags
yourself. Add `--denormalize-precomputed-scores` if SNPMI is stored on `[0, 1]`.

## Scripts

### `scripts/compute.sh`

Wraps `score_lm_embeddings.py`. Defaults read datasets from
`lm-embedding/datasets/` and write outputs to `lm-embedding/outputs/lm_embeddings/`.
CSV files are preferred when present so row order matches the legacy
precomputed SNPMI score files; `.pkl` files are still supported as a fallback.

Env vars (override on the command line):

```bash
HF_TOKEN=hf_...                       # required only for gated models
DATA_DIR=datasets                     # relative to lm-embedding/
OUTPUT_DIR=outputs/lm_embeddings
SNPMI_DIR=precomputed/snpmi           # auto-discovered SNPMI score CSVs
MATCHA_DIR=precomputed/matcha         # auto-discovered MATCHA score CSVs
DATASETS="snli multi_nli truthfulqa climate_fever coco-caption newts"
MODELS="mpnet s-bert bge-large"       # unset = run all registered models
BATCH_SIZE=64
DEVICE=cuda:0
MAX_LENGTH=512
SKIP_EXISTING=1                       # skip per-model CSVs that already exist
SKIP_AUTO_SNPMI=1                     # disable SNPMI auto-discovery
SKIP_AUTO_MATCHA=1                    # disable MATCHA auto-discovery
SKIP_NLTK_DATA=1                      # do not auto-download punkt/punkt_tab
FAIL_FAST=1                           # stop on first model/import failure
```

### `scripts/analyze.sh`

Wraps `plot_lm_embeddings.py`. Defaults read scores from
`lm-embedding/outputs/lm_embeddings/` and write figures to that folder's
`figures/` subdirectory. This script does not choose models itself; it plots
the models already present in `SCORES_DIR`, usually from the previous
`compute.sh` run.

```bash
SCORES_DIR=outputs/lm_embeddings
FIGURES_DIR=outputs/lm_embeddings/figures
FIG3_DATASETS="snli multi_nli truthfulqa"
FIG5_DATASETS="climate_fever coco-caption newts"
FIG6_DATASETS="snli multi_nli truthfulqa"
```

## HF_TOKEN

`HF_TOKEN` is required only when the requested model set includes gated Hugging
Face models. If `MODELS` is unset, `score_lm_embeddings.py` runs every
registered model, including `mistral-7b`, `llama-2-13b`, and
`llama-3.1-8B-Instruct`. Without a token, those gated models are skipped and
the remaining models continue.

Get a token at <https://huggingface.co/settings/tokens> after accepting the
license for each gated model.

Public-only runs do not need a token:

```bash
MODELS="mpnet s-bert bge-large" lm-embedding/scripts/compute.sh
```

## Models

| Key | Gated |
|---|---|
| `word2vec`, `glove` | no (gensim, no HF) |
| `bert`, `s-bert`, `mpnet`, `distilbert`, `snet-t5`, `e5-large-v2`, `bge-large`, `gte-large` | no |
| `mistral-7b`, `llama-2-13b`, `llama-3.1-8B-Instruct` | **yes** |
| `e5-mistral-7b-instruct`, `speed-embedding-7b-instruct`, `sfr-mistral`, `ling-mistral`, `multilingual-e5-large-instruct`, `jasper`, `stella`, `bilingual-embedding-large`, `jina-embeddings-v3` | no |

Word2Vec / GloVe also need NLTK tokenizer data:

```bash
python3 -m nltk.downloader punkt punkt_tab
```

This data cannot be installed through `requirements.txt`; `requirements.txt`
only installs the `nltk` Python package. `compute.sh` passes
`--ensure-nltk-data` automatically when Word2Vec/GloVe are included, unless
`SKIP_NLTK_DATA=1` is set.

## Layout

```text
lm-embedding/
├── datasets/                 # prepared triplet .csv or .pkl files
├── precomputed/
│   ├── matcha/               # copied MATCHA score CSVs
│   └── snpmi/                # imported SNPMI score CSVs
├── outputs/lm_embeddings/    # scoring outputs (created on run)
├── scripts/
│   ├── compute.sh
│   └── analyze.sh
└── *.py                      # scorer, plotter, model registry, IO helpers
```

Datasets can be CSV files with `premise`, `correct`, `incorrect` columns, or
pickle files with `premise`, `correct_answer`, `incorrect_answer` columns.
The default CSV names are `snli.csv`, `multi_nli.csv`,
`truthfulqa_filtered.csv`, `climate_fever_150.csv`,
`coco-caption-concat.csv`, and `newts_random_first1sent.csv`. CSV rows with
only `correct` or only `incorrect` are kept and scored on the available side;
pandas summaries ignore the missing side. Row-level `gap` is computed only
when both sides exist on the same row, so one-sided datasets such as
Climate-FEVER can show correct/incorrect means without a visible gap bar.
SNPMI files must have `pos_sim` and `neg_sim` columns; `row_id` is used for
alignment if present, otherwise rows align by order.

Outputs:

```text
outputs/lm_embeddings/
├── summary_all.csv
├── failures.csv              # present only if a model/import failed or was skipped
├── {dataset}/
│   ├── scores_long.csv       # main plotting input
│   ├── summary.csv
│   └── model_scores/{model}.csv
└── figures/                  # written by analyze.sh
    ├── summary_table.csv
    ├── embedding_barplots_new/
    └── threshold_curves/
```

## Direct Python Usage

The shell scripts are wrappers. To call the scorer directly:

```bash
cd lm-embedding
python3 score_lm_embeddings.py \
  --dataset snli=datasets/snli.csv \
  --models mpnet s-bert bge-large \
  --precomputed-score snli:snpmi=precomputed/snpmi/snli_snpmi_scores.csv \
  --precomputed-score snli:matcha=precomputed/matcha/snli_matcha_scores.csv \
  --ensure-nltk-data \
  --output-dir outputs/lm_embeddings

python3 plot_lm_embeddings.py \
  --scores-dir outputs/lm_embeddings \
  --output-dir outputs/lm_embeddings/figures
```
