#!/usr/bin/env bash
# Évalue les modèles dont les CSVs se trouvent dans comparison_results/csv_mcq_new/
# Suit la même logique que compare_systems.sh :
#   - Skip si déjà marqué complété dans le state file
#   - Backup du CSV évalué dans comparison_results/csv_eval_new/
#   - Distribution écrite dans comparison_results/distribution_new.output
#
# Usage :
#   ./eval_new.sh                          # tous les modèles disponibles
#   ./eval_new.sh gemma2_9b medGemma_4b   # modèles spécifiques

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$ROOT_DIR/.env"
RESULTS_DIR="$SCRIPT_DIR/comparison_results"
MCQ_NEW_DIR="$RESULTS_DIR/csv_mcq_new"
EVAL_BAK_DIR="$RESULTS_DIR/csv_eval_new"
STATE_FILE="$RESULTS_DIR/state_eval_new.txt"
DIST_FILE="$RESULTS_DIR/distribution_new.output"

# Charger le .env
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

mkdir -p "$EVAL_BAK_DIR"

# Modèles à évaluer : depuis les args ou depuis les CSV disponibles
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=()
    for f in "$MCQ_NEW_DIR"/*.csv; do
        [ -f "$f" ] || continue
        MODELS+=("$(basename "$f" .csv)")
    done
fi

echo ""
echo "========================================"
echo "  Modèles : ${MODELS[*]}"
echo "  Source  : $MCQ_NEW_DIR"
echo "========================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo "════════════════════════════════════════"
    echo "  Modèle : $model"
    echo "════════════════════════════════════════"

    # Skip si déjà complété
    if [ -f "$STATE_FILE" ] && grep -qx "$model" "$STATE_FILE"; then
        echo "[skip] $model déjà évalué"
        continue
    fi

    # Vérifier que le CSV source existe
    if [ ! -f "$MCQ_NEW_DIR/${model}.csv" ]; then
        echo "[WARN] CSV introuvable : $MCQ_NEW_DIR/${model}.csv — on passe"
        continue
    fi

    # Supprimer l'ancien CSV évalué pour forcer la régénération
    [ -n "${MODEL_MCQ_EVAL_EXPORT_PATH:-}" ] && rm -f "${MODEL_MCQ_EVAL_EXPORT_PATH}/${model}.csv"

    # Évaluer — lire depuis csv_mcq_new, écrire distribution dans distribution_new.output
    export MODEL_MCQ_PATH="$MCQ_NEW_DIR"
    export DISTRIBUTION_OUTPUT="$DIST_FILE"

    echo "  → Évaluation..."
    cd "$SCRIPT_DIR"
    ./main.py "$model" || {
        echo "[WARN] Évaluation échouée pour $model, on continue."
        continue
    }

    # Backup du CSV évalué
    if [ -n "${MODEL_MCQ_EVAL_EXPORT_PATH:-}" ] && [ -f "${MODEL_MCQ_EVAL_EXPORT_PATH}/${model}.csv" ]; then
        cp "${MODEL_MCQ_EVAL_EXPORT_PATH}/${model}.csv" "$EVAL_BAK_DIR/${model}.csv"
        echo "  ✓ Backup : $EVAL_BAK_DIR/${model}.csv"
    fi

    # Marquer comme complété
    echo "$model" >> "$STATE_FILE"
    echo "  ✓ $model terminé"
done

echo ""
echo "========================================"
echo "  Distribution : $DIST_FILE"
echo "========================================"
[ -f "$DIST_FILE" ] && cat "$DIST_FILE" || echo "[WARN] Fichier de distribution absent."
