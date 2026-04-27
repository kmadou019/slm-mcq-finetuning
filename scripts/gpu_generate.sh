#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

source /home/daisy/konema/Documents/partages/.venv/bin/activate

"$ROOT_DIR/notebooks/generate_mcq.py"
