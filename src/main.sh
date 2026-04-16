#!/usr/bin/env bash
source ~/Documents/.venv/bin/activate
ollama serve &
mapfile -t models < <(jq -r 'keys[]' ../data/models.json)

for model in "${models[@]}";
do
    ../notebooks/generate_mcq.py $model
    ./main.py $model
    cp ../data/dataset_with_quality/$model.csv page/backend/data/mcqs
done


