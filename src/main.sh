#!/usr/bin/env bash

models=(
        "llama3_1_8b"
        #"openbiollm_8b"
        #"gemma2_9b"
        #"medGemma_4b"
        #"medGemma_27b"
        #"qwen3_8b"
        #"mistral_7b"
        #"eurollm_9b"
        #"apertus_8B" 
        #"qwen3_0.6b"
        #"qwen3_1_7b"
        #"qwen3_4b"
        #"qwen3_8b_pdapt_slerp"
        #"qwen3_4b_pdapt_slerp"
        #"qwen3_1_7b_pdapt_slerp",
        #"qwen3_0.6b_pdapt_slerp"
)

for model in "${models[@]}";
do
    ./main.py $model
    cp ../data/dataset_with_quality/$model.csv page/backend/data/mcqs
done


