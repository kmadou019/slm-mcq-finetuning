#!/usr/bin/env bash
#OAR -n optimize_prompt
#OAR -p host='lig-gpu10.imag.fr'
#OAR -l /gpu=1,walltime=50:0:0
#OAR -O /home/daisy/konema/Documents/partages/slm-mcq-finetuning/logs/oar_optimize.%jobid%.stdout
#OAR -E /home/daisy/konema/Documents/partages/slm-mcq-finetuning/logs/oar_optimize.%jobid%.stderr

set -e

export PATH="$HOME/.local/bin:$PATH"
source /home/daisy/konema/Documents/partages/.venv/bin/activate
cd "$(dirname "$0")/.."

echo "════════════════════════════════════════"
echo "  Job OAR : optimize_prompt"
echo "  Date    : $(date)"
echo "  Noeud   : $(hostname)"
echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "════════════════════════════════════════"

python scripts/optimize_prompt.py \
    --k 20                        \
    --max-attempts 5

echo "════════════════════════════════════════"
echo "  Terminé : $(date)"
echo "════════════════════════════════════════"
