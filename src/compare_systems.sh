#!/usr/bin/env bash
# Usage: ./compare_systems.sh [model1 model2 ...]
# Sans argument : utilise tous les modèles de main.sh
#
# Lance le pipeline génération+évaluation pour deux systèmes de prompt :
#   - old : prompt fallback générique (sans GRAPHDB)
#   - new : prompt enrichi via retrieval GraphDB
# Les données sont effacées entre les deux runs pour éviter toute contamination.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$ROOT_DIR/.env"
RESULTS_DIR="$SCRIPT_DIR/comparison_results"

# Charger les variables d'environnement de base (OPENAI_KEY, paths...)
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

source ~/Documents/partages/.venv/bin/activate

# Démarrer ollama s'il ne tourne pas déjà
if ! pgrep -x ollama > /dev/null; then
    ollama serve &
    sleep 3
fi

ALL_MODELS=(
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

if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${ALL_MODELS[@]}")
fi

mkdir -p "$RESULTS_DIR"

run_system() {
    local system_name="$1"

    echo ""
    echo "========================================"
    echo "  Système : $system_name"
    echo "========================================"

    # Réinitialiser distribution.output pour ce run
    > "$SCRIPT_DIR/distribution.output"

    # Supprimer les CSVs des modèles testés pour forcer la régénération
    for model in "${MODELS[@]}"; do
        if [ -n "${MODEL_MCQ_PATH:-}" ]; then
            rm -f "${MODEL_MCQ_PATH}/${model}.csv"
        fi
        if [ -n "${MODEL_MCQ_EVAL_EXPORT_PATH:-}" ]; then
            rm -f "${MODEL_MCQ_EVAL_EXPORT_PATH}/${model}.csv"
        fi
    done

    # Générer puis évaluer chaque modèle
    for model in "${MODELS[@]}"; do
        echo ""
        echo "--- [$system_name] Génération : $model ---"
        "$ROOT_DIR/notebooks/generate_mcq.py" "$model" || {
            echo "[WARN] Génération échouée pour $model, on continue."
            continue
        }

        echo "--- [$system_name] Évaluation : $model ---"
        cd "$SCRIPT_DIR"
        ./main.py "$model" || {
            echo "[WARN] Évaluation échouée pour $model, on continue."
            continue
        }
    done

    cp "$SCRIPT_DIR/distribution.output" "$RESULTS_DIR/distribution_${system_name}.output"
    echo ""
    echo "Résultats sauvegardés : $RESULTS_DIR/distribution_${system_name}.output"
}

# =============================================
# Système ANCIEN — fallback (sans GraphDB)
# =============================================
export GRAPHDB_MCP_URL=""
export GRAPHDB_BEARER_TOKEN=""
run_system "old"

# =============================================
# Système NOUVEAU — retrieval via GraphDB
# =============================================
# Recharger les credentials GraphDB depuis .env
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
run_system "new"

# =============================================
# Affichage comparatif
# =============================================
echo ""
echo "========================================"
echo "  Résultats comparatifs"
echo "========================================"
python3 "$SCRIPT_DIR/compare_results.py" \
    "$RESULTS_DIR/distribution_old.output" \
    "$RESULTS_DIR/distribution_new.output"
