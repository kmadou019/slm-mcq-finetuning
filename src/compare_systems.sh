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
# Pour chaque modèle, lance old puis new (isolation garantie).
# Les CSVs sont sauvegardés dans comparison_results/csv_mcq_old/ etc.
# Reprise automatique : les modèles déjà complétés sont sautés.

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

# Auto-détection du venv
if [ -f ~/Documents/partages/.venv/bin/activate ]; then
    source ~/Documents/partages/.venv/bin/activate
elif [ -f ~/Documents/.venv/bin/activate ]; then
    source ~/Documents/.venv/bin/activate
else
    echo "[WARN] Aucun venv trouvé, on continue sans activation"
fi

# Démarrer ollama s'il ne tourne pas déjà
if ! pgrep -x ollama > /dev/null; then
    ollama serve &
    sleep 3
fi

mapfile -t ALL_MODELS < <(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('\n'.join(d.keys()))" "$ROOT_DIR/data/models.json")

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

# ─── run_one_model(system_name, model) ───────────────────────────────────────
# Génère + évalue un modèle avec un système de prompt donné.
# - Skip si déjà marqué complété dans le state file
# - Checkpoints par modèle (start_{model} / df_in_construction_{model}.csv)
# - Sauvegarde les CSVs dans csv_mcq_{system} / csv_eval_{system}
# ─────────────────────────────────────────────────────────────────────────────
run_one_model() {
    local system_name="$1"
    local model="$2"
    local state_file="$RESULTS_DIR/state_${system_name}.txt"
    local dist_file="$RESULTS_DIR/distribution_${system_name}.output"

    # Skip si déjà complété
    if [ -f "$state_file" ] && grep -qx "$model" "$state_file"; then
        echo "[skip] $model/$system_name déjà complété"
        return 0
    fi

    echo ""
    echo "--- [$system_name] Modèle : $model ---"

    # Supprimer les CSVs de travail pour forcer la régénération
    [ -n "${MODEL_MCQ_PATH:-}" ]             && rm -f "${MODEL_MCQ_PATH}/${model}.csv"
    [ -n "${MODEL_MCQ_EVAL_EXPORT_PATH:-}" ] && rm -f "${MODEL_MCQ_EVAL_EXPORT_PATH}/${model}.csv"

    # Générer
    export PROMPT_SYSTEM="$system_name"
    echo "  → Génération..."
    "$ROOT_DIR/notebooks/generate_mcq.py" "$model" "${INDEX_ARGS[@]}" || {
        echo "[WARN] Génération échouée pour $model/$system_name, on continue."
        return 0
    }

    # Sauvegarder le CSV généré immédiatement (avant eval, pour éviter écrasement)
    local mcq_bak="$RESULTS_DIR/csv_mcq_${system_name}"
    local eval_bak="$RESULTS_DIR/csv_eval_${system_name}"
    mkdir -p "$mcq_bak" "$eval_bak"
    [ -f "${MODEL_MCQ_PATH}/${model}.csv" ] && \
        cp "${MODEL_MCQ_PATH}/${model}.csv" "$mcq_bak/${model}.csv"

    # Évaluer — sortie directement dans le fichier distribution du bon système
    export DISTRIBUTION_OUTPUT="$dist_file"
    echo "  → Évaluation..."
    cd "$SCRIPT_DIR"
    ./main.py "$model" || {
        echo "[WARN] Évaluation échouée pour $model/$system_name, on continue."
        return 0
    }

    # Sauvegarder le CSV évalué
    [ -f "${MODEL_MCQ_EVAL_EXPORT_PATH}/${model}.csv" ] && \
        cp "${MODEL_MCQ_EVAL_EXPORT_PATH}/${model}.csv" "$eval_bak/${model}.csv"

    # Marquer comme complété
    echo "$model" >> "$state_file"
    echo "  ✓ $model/$system_name terminé"
}

# =============================================
# Boucle principale : par modèle, old puis new
# =============================================
echo ""
echo "========================================"
echo "  Modèles : ${MODELS[*]}"
echo "========================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo "════════════════════════════════════════"
    echo "  Modèle : $model"
    echo "════════════════════════════════════════"

    run_one_model "old" "$model"
    run_one_model "new" "$model"
done

# =============================================
# Affichage comparatif
# =============================================
echo ""
echo "========================================"
echo "  Résultats comparatifs"
echo "========================================"

OLD_DIST="$RESULTS_DIR/distribution_old.output"
NEW_DIST="$RESULTS_DIR/distribution_new.output"

if [ -f "$OLD_DIST" ] && [ -f "$NEW_DIST" ]; then
    python3 "$SCRIPT_DIR/compare_results.py" "$OLD_DIST" "$NEW_DIST"
else
    echo "[WARN] Fichiers de distribution manquants, pas de comparaison possible."
    [ -f "$OLD_DIST" ] && echo "  old : OK" || echo "  old : manquant"
    [ -f "$NEW_DIST" ] && echo "  new : OK" || echo "  new : manquant"
fi
