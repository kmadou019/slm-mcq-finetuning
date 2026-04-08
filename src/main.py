#!/usr/bin/env python3
from sys import argv

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
            # Chaque paramètre suivant est un index ou un intervalle
            for i in range(2, len(argv)):
                arg = argv[i]

                # Vérifier si c'est un intervalle (contient '-')
                if '-' in arg:
                    parts = arg.split('-')
                    if len(parts) != 2:
                        print(f"Error: Invalid interval format '{arg}'. Use format: start-end (e.g., 1-10)")
                        exit(1)

                    start = int(parts[0])
                    end = int(parts[1])

                    if start > end:
                        print(f"Error: Invalid interval '{arg}'. Start must be <= end.")
                        exit(1)

                    # Ajouter tous les index de l'intervalle
                    indexes.extend(range(start, end + 1))
                else:
                    # C'est un index simple
                    index = int(arg)
                    indexes.append(index)

            print(f"✓ Model: {model}")
            print(f"✓ Generated QCM file: {generated_qcm_file}")
            print(f"✓ Indexes: {indexes}")

        except ValueError as e:
            print(f"Error: Invalid index/interval at position {i}. All indexes must be integers.")
            print(f"Details: {e}")
            exit(1)
    else:
        print(f"✓ Model: {model}")
        print(f"✓ Generated QCM file: {generated_qcm_file}")
        print("⚠ No indexes provided (optional)")

else:
    print("Error: Missing model name")
    print("\nUsage: ./code.py <model> [index1] [index2] [start-end] ...")
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
    print("  ./main.py llama3_1_8b")
    print("  ./main.py llama3_1_8b 1 2 3")
    print("  ./main.py llama3_1_8b 1-10")
    print("  ./main.py openbiollm_8b 0-5 8 10-15")
    print("  ./main.py gemma2_9b 0 5 10 15")
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
    df_lisa_sheets = df_lisa_sheets[df_lisa_sheets['id'].isin(df_mcq["id"])].iloc[:1]
    df_mcq = df_mcq[df_mcq["id"].isin(df_lisa_sheets["id"])]

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
    

    df_eval.to_csv(os.environ.get('MODEL_MCQ_EVAL_EXPORT_PATH') + "/" +  generated_qcm_file, index=False)

    with open("distribution.output", "a") as f:
        print(f"\n{generated_qcm_file} quality distribution:", file=f)
        total = len(df_eval)
        counts = df_eval['distractor_quality'].round(0).value_counts(sort=True)
        for val, count in counts.items():
            pct = count / total * 100
            label = "Bonne qualité" if val else "Mauvaise qualité"
            print(f"  {label} ({val}): {pct:.2f}% (n={count}/{total})", file=f)


if __name__ == '__main__':
    main()
