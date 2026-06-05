#!/usr/bin/env bash
#OAR -n compare_optimized
#OAR -p host='lig-gpu10.imag.fr'
#OAR -l /gpu=1,walltime=20:0:0
#OAR -O /home/daisy/konema/Documents/partages/slm-mcq-finetuning/logs/oar_compare_optimized.%jobid%.stdout
#OAR -E /home/daisy/konema/Documents/partages/slm-mcq-finetuning/logs/oar_compare_optimized.%jobid%.stderr

set -e

source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."

mkdir -p logs

echo "════════════════════════════════════════"
echo "  Job OAR : compare_optimized"
echo "  Expérience : optimized vs optimized_enriched"
echo "  Modèles    : qwen3_0_6b, llama3_8b"
echo "  Date       : $(date)"
echo "  Noeud      : $(hostname)"
echo "  GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "════════════════════════════════════════"

bash scripts/compare_systems.sh qwen3_0_6b llama3_8b

echo "════════════════════════════════════════"
echo "  Terminé : $(date)"
echo "════════════════════════════════════════"
