#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_EMBEDDING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

resolve_path() {
    local base="$1"
    local path="$2"
    case "$path" in
        /*) printf '%s\n' "$path" ;;
        *) printf '%s/%s\n' "$base" "$path" ;;
    esac
}

dataset_path() {
    local dataset="$1"
    local csv_name

    case "$dataset" in
        truthfulqa) csv_name="truthfulqa_filtered.csv" ;;
        climate_fever) csv_name="climate_fever_150.csv" ;;
        coco-caption) csv_name="coco-caption-concat.csv" ;;
        newts) csv_name="newts_random_first1sent.csv" ;;
        *) csv_name="$dataset.csv" ;;
    esac

    local candidate
    for candidate in "$DATA_DIR/$csv_name" "$DATA_DIR/$dataset.pkl"; do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

requires_hf_token() {
    # MODELS unset means score_lm_embeddings.py will run every registered model,
    # including the gated Hugging Face models below.
    if [[ -z "${MODELS:-}" ]]; then
        return 0
    fi

    local model
    for model in $MODELS; do
        case "$model" in
            mistral-7b|llama-2-13b|llama-3.1-8B-Instruct) return 0 ;;
        esac
    done
    return 1
}

DATA_DIR="$(resolve_path "$LM_EMBEDDING_DIR" "${DATA_DIR:-data}")"
OUTPUT_DIR="$(resolve_path "$LM_EMBEDDING_DIR" "${OUTPUT_DIR:-outputs/lm_embeddings}")"
SNPMI_DIR="$(resolve_path "$LM_EMBEDDING_DIR" "${SNPMI_DIR:-precomputed/snpmi}")"
DATASETS="${DATASETS:-snli multi_nli truthfulqa climate_fever coco-caption newts}"

if requires_hf_token && [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set, but the requested model set includes gated Hugging Face models." >&2
    echo "Gated models: mistral-7b, llama-2-13b, llama-3.1-8B-Instruct" >&2
    echo "Export HF_TOKEN=hf_..., or set MODELS to public/non-gated models only." >&2
    exit 1
fi

# Uncomment and edit this line to choose specific models.
# If MODELS is left unset, score_lm_embeddings.py uses all registered models.
#
# MODELS="mpnet s-bert bge-large"
#
# Available models:
# word2vec
# glove
# bert
# s-bert
# mpnet
# distilbert
# snet-t5
# e5-large-v2
# bge-large
# gte-large
# mistral-7b
# llama-2-13b
# llama-3.1-8B-Instruct
# e5-mistral-7b-instruct
# speed-embedding-7b-instruct
# sfr-mistral
# ling-mistral
# multilingual-e5-large-instruct
# jasper
# stella
# bilingual-embedding-large
# jina-embeddings-v3


DATASET_ARGS=()
PRECOMPUTED_ARGS=()
for dataset in $DATASETS; do
    if ! path="$(dataset_path "$dataset")"; then
        echo "Missing dataset file for: $dataset" >&2
        echo "Set DATA_DIR to the folder containing the prepared .csv or .pkl files" >&2
        exit 1
    fi
    DATASET_ARGS+=(--dataset "$dataset=$path")

    # Include SNPMI scores only from the exact repo-owned naming convention.
    # Set SKIP_AUTO_SNPMI=1 to disable (e.g. when passing --precomputed-score manually).
    if [[ "${SKIP_AUTO_SNPMI:-0}" != "1" ]]; then
        snpmi_path="$SNPMI_DIR/${dataset}_snpmi_scores.csv"
        if [[ -f "$snpmi_path" ]]; then
            PRECOMPUTED_ARGS+=(--precomputed-score "$dataset:snpmi=$snpmi_path")
        fi
    fi
done

MODEL_ARGS=()
if [[ -n "${MODELS:-}" ]]; then
    read -r -a MODEL_LIST <<< "$MODELS"
    MODEL_ARGS=(--models "${MODEL_LIST[@]}")
fi

REQUIRED_ARGS=()
[[ -n "${HF_TOKEN:-}" ]] && REQUIRED_ARGS+=(--hf-token "$HF_TOKEN")

OPTIONAL_ARGS=()
[[ -n "${BATCH_SIZE:-}" ]] && OPTIONAL_ARGS+=(--batch-size "$BATCH_SIZE")
[[ -n "${DEVICE:-}" ]] && OPTIONAL_ARGS+=(--device "$DEVICE")
[[ -n "${MAX_LENGTH:-}" ]] && OPTIONAL_ARGS+=(--max-length "$MAX_LENGTH")
[[ "${SKIP_EXISTING:-}" == "1" ]] && OPTIONAL_ARGS+=(--skip-existing)

cd "$LM_EMBEDDING_DIR"

"$PYTHON_BIN" "$LM_EMBEDDING_DIR/score_lm_embeddings.py" \
    "${DATASET_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${PRECOMPUTED_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    "${REQUIRED_ARGS[@]}" \
    "${OPTIONAL_ARGS[@]}" \
    "$@"
