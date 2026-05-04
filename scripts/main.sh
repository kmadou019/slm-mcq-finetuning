#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
ollama serve &
mapfile -t models < <(jq -r 'keys[]' ../data/models.json)

for model in "${models[@]}";
do
    ../notebooks/generate_mcq.py $model
    ../src/main.py $model
    cp ../data/dataset_with_quality/$model.csv ../src/page/backend/data/mcqs
done


