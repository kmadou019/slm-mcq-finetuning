#!/usr/bin/env bash
set -e
source /home/daisy/konema/Documents/partages/.venv/bin/activate
cd "$(dirname "$0")"

N_GPU=1

# Format : "hf_model_id|output_dir|batch_size|grad_acc|lr"

# ─────────────────────────────────────────────
# Phase 1 — Modèles ≤4B
# ─────────────────────────────────────────────
MODELS_SMALL=(
    "Qwen/Qwen3-0.6B|finetuned_models/qwen3-0.6b-KTO|8|2|5e-7"
    "Qwen/Qwen3-1.7B|finetuned_models/qwen3-1.7b-KTO|8|2|5e-7"
    "Qwen/Qwen3-4B|finetuned_models/qwen3-4b-KTO|8|2|5e-7"
    "google/medgemma-4b-it|finetuned_models/medgemma-4b-KTO|8|2|5e-7"
)

# ─────────────────────────────────────────────
# Phase 2 — Modèles ~7-9B
# ─────────────────────────────────────────────
MODELS_MEDIUM=(
    "mistralai/Mistral-7B-Instruct-v0.3|finetuned_models/mistral-7b-KTO|4|4|5e-7"
    "meta-llama/Llama-3.1-8B-Instruct|finetuned_models/llama3-8b-KTO|4|4|5e-7"
    "antonkirk/Llama3-Instruct-OpenBioLLM-8B-merged|finetuned_models/openbiollm-8b-KTO|4|4|5e-7"
    "swiss-ai/Apertus-8B-Instruct-2509|finetuned_models/apertus-8b-KTO|4|4|5e-7"
    "google/gemma-2-9b-it|finetuned_models/gemma2-9b-KTO|4|4|5e-7"
    "utter-project/EuroLLM-9B-Instruct|finetuned_models/eurollm-9b-KTO|4|4|5e-7"
)

# ─────────────────────────────────────────────
# Phase 3 — Modèles 27B+  (en dernier)
# ─────────────────────────────────────────────
MODELS_LARGE=(
    "google/medgemma-27b-it|finetuned_models/medgemma-27b-KTO|1|16|2e-7"
    "google/gemma-4-31B-it|finetuned_models/gemma4-31b-KTO|1|16|2e-7"
    "mistralai/Mixtral-8x7B-Instruct-v0.1|finetuned_models/mixtral-8x7b-KTO|1|16|2e-7"
    "mistralai/Magistral-Small-2509|finetuned_models/magistral-small-KTO|1|16|2e-7"
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16|finetuned_models/nemotron-30b-KTO|1|16|2e-7"
    "Qwen/Qwen3.5-35B-A3B|finetuned_models/qwen3.5-35b-KTO|1|16|2e-7"
)

finetune() {
    IFS='|' read -r model outdir batch grad_acc lr <<< "$1"
    echo ""
    echo "════════════════════════════════════════"
    echo "  Modèle     : $model"
    echo "  Output     : $outdir"
    echo "  Batch/GPU  : $batch   Grad acc : $grad_acc   LR : $lr"
    echo "════════════════════════════════════════"
    accelerate launch --num_processes=$N_GPU ../notebooks/kto_training.py \
        --model      "$model"    \
        --output-dir "$outdir"   \
        --batch-size "$batch"    \
        --grad-acc   "$grad_acc" \
        --lr         "$lr"
    echo "✓ Terminé : $model"
}

echo "══════════════════════════════════════════"
echo "  Phase 1 — Modèles ≤4B ($N_GPU GPUs)"
echo "══════════════════════════════════════════"
for entry in "${MODELS_SMALL[@]}"; do
    finetune "$entry"
done

echo ""
echo "══════════════════════════════════════════"
echo "  Phase 2 — Modèles ~7-9B ($N_GPU GPUs)"
echo "══════════════════════════════════════════"
for entry in "${MODELS_MEDIUM[@]}"; do
    finetune "$entry"
done

echo ""
echo "══════════════════════════════════════════"
echo "  Phase 3 — Modèles 27B+ ($N_GPU GPUs)"
echo "══════════════════════════════════════════"
for entry in "${MODELS_LARGE[@]}"; do
    finetune "$entry"
done

echo ""
echo "✓ Finetuning complet — $(date)"
