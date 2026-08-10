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

SCORES_DIR="$(resolve_path "$LM_EMBEDDING_DIR" "${SCORES_DIR:-outputs/lm_embeddings}")"
FIGURES_DIR="$(resolve_path "$LM_EMBEDDING_DIR" "${FIGURES_DIR:-outputs/lm_embeddings/figures}")"

OPTIONAL_ARGS=()
if [[ -n "${FIG3_DATASETS:-}" ]]; then
    read -r -a FIG3_LIST <<< "$FIG3_DATASETS"
    OPTIONAL_ARGS+=(--fig3-datasets "${FIG3_LIST[@]}")
fi
if [[ -n "${FIG5_DATASETS:-}" ]]; then
    read -r -a FIG5_LIST <<< "$FIG5_DATASETS"
    OPTIONAL_ARGS+=(--fig5-datasets "${FIG5_LIST[@]}")
fi
if [[ -n "${FIG6_DATASETS:-}" ]]; then
    read -r -a FIG6_LIST <<< "$FIG6_DATASETS"
    OPTIONAL_ARGS+=(--fig6-datasets "${FIG6_LIST[@]}")
fi

cd "$LM_EMBEDDING_DIR"

"$PYTHON_BIN" "$LM_EMBEDDING_DIR/plot_lm_embeddings.py" \
    --scores-dir "$SCORES_DIR" \
    --output-dir "$FIGURES_DIR" \
    "${OPTIONAL_ARGS[@]}" \
    "$@"
