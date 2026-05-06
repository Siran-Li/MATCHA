# LM Embedding Baselines
Compare different language model embedding models, testing their ability to distinguish between correct and incorrect pairs using cosine similarity.

Compute and plot embedding-based semantic-separation scores.

## Quick Start

```bash
# 1. install deps
pip install -r lm-embedding/requirements.txt

# 2. export your Hugging Face token (required, see "HF_TOKEN" below)
export HF_TOKEN=hf_...

# 3. score embeddings
lm-embedding/scripts/compute.sh

# 4. plot figures
lm-embedding/scripts/analyze.sh
```

Prerequisite: prepared dataset files must live under `lm-embedding/data/` with
names matching the dataset keys, for example `snli.pkl`, `multi_nli.pkl`,
`truthfulqa.pkl`, `climate_fever.pkl`, `coco-caption.pkl`, `newts.pkl`.
# TODO
refer to the prepare_eval_datasets.py with the validation split

SNPMI scores are auto-included: any file matching
`precomputed_snpmi/{dataset}_snpmi_scores.csv` is picked up automatically for
the datasets in `DATASETS`. To override, set `SNPMI_DIR=/some/path`, or set
`SKIP_AUTO_SNPMI=1` and pass `--precomputed-score` flags yourself. Add
`--denormalize-precomputed-scores` if SNPMI is stored on `[0, 1]`.

## Scripts

### `scripts/compute.sh`

Wraps `score_lm_embeddings.py`. Defaults read datasets from
`lm-embedding/data/` and write outputs to `lm-embedding/outputs/lm_embeddings/`.

Env vars (override on the command line):

```bash
HF_TOKEN=hf_...                       # required, see below
DATA_DIR=data                         # relative to lm-embedding/
OUTPUT_DIR=outputs/lm_embeddings
SNPMI_DIR=precomputed_snpmi           # auto-discovered SNPMI score CSVs
DATASETS="snli multi_nli truthfulqa climate_fever coco-caption newts"
MODELS="mpnet s-bert bge-large"       # unset = run all registered models
BATCH_SIZE=64
DEVICE=cuda:0
MAX_LENGTH=512
SKIP_EXISTING=1                       # skip per-model CSVs that already exist
SKIP_AUTO_SNPMI=1                     # disable SNPMI auto-discovery
ALLOW_MISSING_HF_TOKEN=1              # only for word2vec/glove-only runs
```

### `scripts/analyze.sh`

Wraps `plot_lm_embeddings.py`. Defaults read scores from
`lm-embedding/outputs/lm_embeddings/` and write figures to that folder's
`figures/` subdirectory.

```bash
SCORES_DIR=outputs/lm_embeddings
FIGURES_DIR=outputs/lm_embeddings/figures
FIG3_DATASETS="snli multi_nli truthfulqa"
FIG5_DATASETS="climate_fever coco-caption newts"
FIG6_DATASETS="snli multi_nli truthfulqa"
```

## HF_TOKEN (Required)

`HF_TOKEN` is required for any run that touches Hugging Face. Every transformer
and sentence-transformer load path in `load_models.py` reads it via
`os.getenv("HF_TOKEN")` and the default model list contains gated models
(`mistral-7b`, `llama-2-13b`, `llama-3.1-8B-Instruct`) that fail without auth.

Get a token at <https://huggingface.co/settings/tokens> after accepting the
license for each gated model.

`compute.sh` fails fast if `HF_TOKEN` is unset. The only run that doesn't
need it is gensim-only:

```bash
ALLOW_MISSING_HF_TOKEN=1 MODELS="word2vec glove" lm-embedding/scripts/compute.sh
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

## Layout

```text
lm-embedding/
├── data/                     # prepared triplet .pkl files
├── precomputed_snpmi/        # imported SNPMI score CSVs
├── outputs/lm_embeddings/    # scoring outputs (created on run)
├── scripts/
│   ├── compute.sh
│   └── analyze.sh
└── *.py                      # scorer, plotter, model registry, IO helpers
```

Datasets must be `.pkl` files with the columns `premise`, `correct_answer`,
`incorrect_answer`. SNPMI files must have `pos_sim` and `neg_sim` columns;
`row_id` is used for alignment if present, otherwise rows align by order.

Outputs:

```text
outputs/lm_embeddings/
├── summary_all.csv
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
  --dataset snli=data/snli.pkl \
  --models mpnet s-bert bge-large \
  --precomputed-score snli:snpmi=precomputed_snpmi/snli_snpmi_scores.csv \
  --output-dir outputs/lm_embeddings

python3 plot_lm_embeddings.py \
  --scores-dir outputs/lm_embeddings \
  --output-dir outputs/lm_embeddings/figures
```
