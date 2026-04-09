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

# ─── PARAMÈTRE DE TEST ───────────────────────────────────────────────────────
# Mettre un range ici pour limiter les sheets (ex: "0-9"), laisser vide pour tout
TEST_RANGE=""
# ─────────────────────────────────────────────────────────────────────────────

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
    #"openbiollm_8b"
    #"gemma2_9b"
    #"medGemma_4b"
    #"medGemma_27b"
    #"qwen3_8b"
    #"mistral_7b"
    #"eurollm_9b"
    #"apertus_8B"
    "qwen3_0.6b"
    "qwen3_1_7b"
    #"qwen3_4b"
    #"qwen3_8b_pdapt_slerp"
    #"qwen3_4b_pdapt_slerp"
    #"qwen3_1_7b_pdapt_slerp"
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

# TEST_RANGE injecté si aucun index passé en argument
if [ ${#INDEX_ARGS[@]} -eq 0 ] && [ -n "$TEST_RANGE" ]; then
    INDEX_ARGS=("$TEST_RANGE")
fi

if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=("${ALL_MODELS[@]}")
fi

if [ ${#INDEX_ARGS[@]} -gt 0 ]; then
    echo "✓ Subset de sheets : ${INDEX_ARGS[*]}"
fi

mkdir -p "$RESULTS_DIR"

run_system() {
    local system_name="$1"
    local state_file="$RESULTS_DIR/state_${system_name}.txt"

    echo ""
    echo "========================================"
    echo "  Système : $system_name"
    echo "========================================"

    # Charger les modèles déjà complétés
    local done_models=()
    if [ -f "$state_file" ]; then
        mapfile -t done_models < "$state_file"
        echo "[resume] ${#done_models[@]} modèles déjà complétés pour '$system_name' : ${done_models[*]}"
    fi

    # Vider distribution.output seulement si on repart de zéro
    if [ ${#done_models[@]} -eq 0 ]; then
        > "$SCRIPT_DIR/distribution.output"
    fi

    # Supprimer les CSVs uniquement des modèles non encore complétés
    for model in "${MODELS[@]}"; do
        if printf '%s\n' "${done_models[@]}" | grep -qx "$model"; then
            continue
        fi
        [ -n "${MODEL_MCQ_PATH:-}" ]      && rm -f "${MODEL_MCQ_PATH}/${model}.csv"
        [ -n "${MODEL_MCQ_EVAL_EXPORT_PATH:-}" ] && rm -f "${MODEL_MCQ_EVAL_EXPORT_PATH}/${model}.csv"
    done

    # Générer puis évaluer chaque modèle
    for model in "${MODELS[@]}"; do
        # Skip si déjà complété
        if printf '%s\n' "${done_models[@]}" | grep -qx "$model"; then
            echo "[skip] $model déjà complété"
            continue
        fi

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

        # Marquer comme complété
        echo "$model" >> "$state_file"
        done_models+=("$model")
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
