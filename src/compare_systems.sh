#!/usr/bin/env bash
# Usage: ./compare_systems.sh [model1 model2 ...] [index_or_range ...]
#   - Sans argument          : tous les modèles, toutes les sheets
#   - Avec modèles seulement : modèles spécifiés, toutes les sheets
#   - Avec index/ranges      : les args numériques (ex: 0 1-4) filtrent les lisa sheets
#
# Exemples :
#   ./compare_systems.sh qwen3_0.6b 0-4      # 1 modèle, 5 sheets
#   ./compare_systems.sh qwen3_0.6b qwen3_1_7b 0 1 2
#   ./compare_systems.sh                      # run complet
#
# Lance le pipeline génération+évaluation pour deux systèmes de prompt :
#   - old : prompt simple (commit ba23972)
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

# Séparer les modèles (non-numériques) des index/ranges (numériques ou X-Y)
MODELS=()
INDEX_ARGS=()
for arg in "$@"; do
    if [[ "$arg" =~ ^[0-9]+(-[0-9]+)?$ ]]; then
        INDEX_ARGS+=("$arg")
    else
        MODELS+=("$arg")
    fi
done

if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=("${ALL_MODELS[@]}")
fi

if [ ${#INDEX_ARGS[@]} -gt 0 ]; then
    echo "✓ Subset de sheets : ${INDEX_ARGS[*]}"
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
        "$ROOT_DIR/notebooks/generate_mcq.py" "$model" "${INDEX_ARGS[@]}" || {
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
# Système ANCIEN — prompt simple (ba23972)
# =============================================
export PROMPT_SYSTEM="old"
run_system "old"

# =============================================
# Système NOUVEAU — retrieval via GraphDB
# =============================================
export PROMPT_SYSTEM="new"
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
