#!/usr/bin/env bash
#OAR -n transfer_qwen
#OAR -p host='lig-gpu10.imag.fr'
#OAR -l /gpu=1,walltime=20:0:0
#OAR -O /home/daisy/konema/Documents/partages/slm-mcq-finetuning/logs/oar_transfer_qwen.%jobid%.stdout
#OAR -E /home/daisy/konema/Documents/partages/slm-mcq-finetuning/logs/oar_transfer_qwen.%jobid%.stderr

set -e

source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."

mkdir -p logs

echo "════════════════════════════════════════"
echo "  Job OAR : transfer_qwen"
echo "  Expérience : prompt qwen3_0_6b → grands modèles"
echo "  Modèles    : medgemma_27b, gemma4_31b"
echo "  Date       : $(date)"
echo "  Noeud      : $(hostname)"
echo "  GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "════════════════════════════════════════"

export PROMPT_SYSTEM="transfer_qwen"

echo ""
echo "── medgemma_27b ────────────────────────"
python notebooks/generate_mcq.py medgemma_27b

echo ""
echo "── gemma4_31b ──────────────────────────"
python notebooks/generate_mcq.py gemma4_31b

echo ""
echo "════════════════════════════════════════"
echo "  Terminé : $(date)"
echo "════════════════════════════════════════"
