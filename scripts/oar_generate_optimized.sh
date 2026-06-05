#!/usr/bin/env bash
#OAR -n generate_optimized
#OAR -p host='lig-gpu10.imag.fr'
#OAR -l /gpu=1,walltime=50:0:0
#OAR -O /home/daisy/konema/Documents/partages/slm-mcq-finetuning/logs/oar_generate_optimized.%jobid%.stdout
#OAR -E /home/daisy/konema/Documents/partages/slm-mcq-finetuning/logs/oar_generate_optimized.%jobid%.stderr

set -e

source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."

mkdir -p logs

echo "════════════════════════════════════════"
echo "  Job OAR : generate_optimized"
echo "  Modèles  : 8 modèles avec prompt optimisé"
echo "  Date     : $(date)"
echo "  Noeud    : $(hostname)"
echo "  GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "════════════════════════════════════════"

bash scripts/compare_systems.sh \
    qwen3_0_6b \
    mistral_7b \
    llama3_8b \
    apertus_8b \
    eurollm_9b \
    gemma2_9b \
    medgemma_27b \
    gemma4_31b

echo "════════════════════════════════════════"
echo "  Terminé : $(date)"
echo "════════════════════════════════════════"
