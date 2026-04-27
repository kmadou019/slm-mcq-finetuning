#!/usr/bin/env bash
# Pipeline génération + évaluation pour tous les modèles de data/models.json
# Usage : ./generate.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$ROOT_DIR/src"

source /home/daisy/konema/Documents/partages/.venv/bin/activate

if ! pgrep -x ollama > /dev/null; then
    ollama serve &
    sleep 3
fi

mapfile -t models < <(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('\n'.join(d.keys()))" "$ROOT_DIR/data/models.json")

for model in "${models[@]}"; do
    "$ROOT_DIR/notebooks/generate_mcq.py" "$model"
    cd "$SRC_DIR" && ./main.py "$model"
    cp "$ROOT_DIR/data/dataset_with_quality/${model}.csv" "$SRC_DIR/page/backend/data/mcqs/"
done
