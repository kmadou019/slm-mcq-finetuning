#!/usr/bin/env bash
# Wrapper oarsub : patche les #OAR -O/-E avec le PROJECT_ROOT courant avant soumission.
# Usage: ./scripts/submit.sh scripts/run_finetuning.sh [oarsub args...]
#
# Les directives #OAR sont statiques (pas d'expansion de variables).
# Ce script génère un fichier temporaire avec les bons chemins puis le soumet.

set -e
source "$(dirname "$0")/env.sh"

if [ -z "$1" ]; then
    echo "Usage: $0 <script.sh> [oarsub args...]" >&2
    exit 1
fi

SCRIPT=$(realpath "$1"); shift
TMP=$(mktemp /tmp/oar_submit_XXXXX.sh)

sed "s|/home/[^/]*/[^/]*/Documents/partages/slm-mcq-finetuning|$PROJECT_ROOT|g" "$SCRIPT" > "$TMP"
chmod +x "$TMP"

echo "[submit] PROJECT_ROOT = $PROJECT_ROOT"
echo "[submit] Script temporaire : $TMP"
oarsub -S "$TMP" "$@"

rm -f "$TMP"
