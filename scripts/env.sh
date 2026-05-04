#!/usr/bin/env bash
# Central environment config — sourced by all scripts.
# Uses $HOME to build paths, works on any machine.

export PROJECT_ROOT="$HOME/Documents/partages/slm-mcq-finetuning"
export PATH="$HOME/.local/bin:$PATH"

if [ -f "$HOME/Documents/partages/.venv/bin/activate" ]; then
    source "$HOME/Documents/partages/.venv/bin/activate"
elif [ -f "$HOME/Documents/.venv/bin/activate" ]; then
    source "$HOME/Documents/.venv/bin/activate"
else
    echo "[env.sh] ERREUR : venv introuvable" >&2
    exit 1
fi
