#!/usr/bin/env python3
from sys import argv
from card import generer_mcq_multi_cartes, construire_params_depuis_csv

if len(argv) >= 2:
    # Vérifier le modèle (premier paramètre)
    valid_models = [
        "llama3_1_8b", "openbiollm_8b", "gemma2_9b", 
        "medGemma_4b", "medGemma_27b", "qwen3_8b", 
        "mistral_7b", "eurollm_9b", "apertus_8B", 
        "qwen3_0.6b", "qwen3_1_7b", "qwen3_4b",
        "qwen3_8b_pdapt_slerp","qwen3_4b_pdapt_slerp",
        "qwen3_1_7b_pdapt_slerp","qwen3_0.6b_pdapt_slerp"
    ]
    
    if argv[1] not in valid_models:
        print(f"Error: Please provide a valid model name:")
        for model in valid_models:
            print(f"  - {model}")
        exit(1)
    
    model = argv[1]
    generated_qcm_file = model + ".csv"
    
    # Extraire les indexes (tous les paramètres après le modèle)
    indexes = []
    if len(argv) >= 3:
        try:
            # Chaque paramètre suivant est un index
            for i in range(2, len(argv)):
                index = int(argv[i])
                indexes.append(index)
            
            print(f"✓ Model: {model}")
            print(f"✓ Generated QCM file: {generated_qcm_file}")
            print(f"✓ Indexes: {indexes}")
        
        except ValueError:
            print(f"Error: Invalid index at position {i}. All indexes must be integers.")
            exit(1)
    else:
        print(f"✓ Model: {model}")
        print(f"✓ Generated QCM file: {generated_qcm_file}")
        print("⚠ No indexes provided (optional)")

else:
    print("Error: Missing model name")
    print("\nUsage: ./code.py <model> [index1] [index2] [index3] ...")
    print("\nValid models:")
    valid_models = [
        "llama3_1_8b", "openbiollm_8b", "gemma2_9b", 
        "medGemma_4b", "medGemma_27b", "qwen3_8b", 
        "mistral_7b", "eurollm_9b", "apertus_8B", 
        "qwen3_0.6b", "qwen3_1_7b", "qwen3_4b"
    ]
    for model in valid_models:
        print(f"  - {model}")
    print("\nExamples:")
    print("  ./code.py llama3_1_8b")
    print("  ./code.py llama3_1_8b 1 2 3")
    print("  ./code.py openbiollm_8b 8 5")
    print("  ./code.py gemma2_9b 0 5 10 15")
    exit(1)

from dotenv import load_dotenv
# disable SSL check to download nltk pakages on MacOS


import os
import json
import pandas as pd
from eval.eval_dataframe import eval_dataframe_parallel
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')



def main():
    load_dotenv()
    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

    with open('eval/prompts.json', 'r') as file:
        # Load the JSON data from the file
        system_prompts = json.load(file)

    df_mcq = pd.read_csv(os.environ.get('MODEL_MCQ_PATH') + "/" + generated_qcm_file)
    if indexes:
        df_mcq = df_mcq.loc[indexes]
    df_lisa_sheets = pd.read_csv(os.environ.get('LISA_SHEETS_PATH'))

    
    # test small subset
    # common_ids = df_mcq['id'].isin(df_lisa_sheets['id'])
    # df_mcq = df_mcq[common_ids].iloc[:60]
    #df_mcq_ids = [idx[:12] for idx in df_mcq['id']]
    df_mcq["id"] = df_mcq["id"].map(lambda idx : idx[:12])
    df_lisa_sheets = df_lisa_sheets[df_lisa_sheets['id'].isin(df_mcq["id"])]

    df_eval = eval_dataframe_parallel(df_mcqs=df_mcq,
                                      df_lisa_sheets=df_lisa_sheets,
                                      openai_key=OPENAI_KEY,
                                      num_workers=10,
                                      lisa_sheet_id_col='id',
                                      lisa_sheet_col='content_raw',
                                      compute_answerability      =False,
                                      compute_originality        =False,
                                      compute_readability        =False,
                                      compute_negation           =False,
                                      compute_is_question        =False,
                                      compute_relevance          =False,
                                      compute_ambiguity          =False,
                                      compute_disclosure         =False,
                                      compute_difficulty         =False,
                                      compute_distractors_quality=True,
                                      disclosure_system_prompt=system_prompts['disclosure_prompt'],
                                      difficulty_system_prompt=system_prompts['difficulty_prompt'],
                                      distractors_quality_system_prompt=system_prompts['distractors_quality_prompt'],
                                      distractors_quality_col='distractor_quality',
                                      merge=True # set to True if your dataframe does not have the Lisa Sheet content
                                      )
    

    params_list = construire_params_depuis_csv(df_eval,model)
    generer_mcq_multi_cartes(params_list, "mcq_cards.html")

    df_eval.to_csv(os.environ.get('MODEL_MCQ_EVAL_EXPORT_PATH') + "/" +  generated_qcm_file, index=False)


if __name__ == '__main__':
    main()
