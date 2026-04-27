#!/usr/bin/env bash
# Pipeline génération + évaluation avec liste de modèles fixe (inclut SLERP)
# Usage : ./generate_full.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$ROOT_DIR/src"

source /home/daisy/konema/Documents/partages/.venv/bin/activate

models=(
    "llama3_1_8b"
    "openbiollm_8b"
    "gemma2_9b"
    "medGemma_4b"
    "medGemma_27b"
    "qwen3_8b"
    "mistral_7b"
    "eurollm_9b"
    "apertus_8B"
    "qwen3_0.6b"
    "qwen3_1_7b"
    "qwen3_4b"
    "qwen3_8b_pdapt_slerp"
    "qwen3_4b_pdapt_slerp"
    "qwen3_1_7b_pdapt_slerp"
    "qwen3_0.6b_pdapt_slerp"
)

for model in "${models[@]}"; do
    "$ROOT_DIR/notebooks/generate_mcq.py" "$model"
    cd "$SRC_DIR" && ./main.py "$model"
    cp "$ROOT_DIR/data/dataset_with_quality/${model}.csv" "$SRC_DIR/page/backend/data/mcqs/"
done
