#!/usr/bin/env python3

from pydantic import BaseModel
import json
import pandas as pd
from pydantic import ValidationError
from pandas import DataFrame
from ollama import generate
from transformers import AutoTokenizer, pipeline
from dotenv import load_dotenv
import os
import ast
import json
from huggingface_hub import login


class MCQQuestion(BaseModel):
    question1: str
    option_a1: str
    option_b1: str
    option_c1: str
    option_d1: str
    correct_option1: str
    question2: str
    option_a2: str
    option_b2: str
    option_c2: str
    option_d2: str
    correct_option2: str

def validate_mcq(mcq_json):
    try:
        return MCQQuestion.model_validate_json(mcq_json)
    except ValidationError as e:
        print(f"Validation failed: {e}")
        return None
        


def flatten_and_export_mcq(df: DataFrame, export_filename: str, mcq_column_name: str):
    ids = [x for val in df["id"] for x in (val, val+'-') ]
    result_df = pd.DataFrame({"id": ids})
    
    result_df['question'] = pd.concat([df[mcq_column_name].apply(lambda x: x.question1 if x else ""), df[mcq_column_name].apply(lambda x: x.question2 if x else "")], ignore_index=True)
    result_df['option_a'] = pd.concat([df[mcq_column_name].apply(lambda x: x.option_a1 if x else ""), df[mcq_column_name].apply(lambda x: x.option_a2 if x else "")], ignore_index=True)
    result_df['option_b'] = pd.concat([df[mcq_column_name].apply(lambda x: x.option_b1 if x else ""), df[mcq_column_name].apply(lambda x: x.option_b2 if x else "")], ignore_index=True) 
    result_df['option_c'] = pd.concat([df[mcq_column_name].apply(lambda x: x.option_c1 if x else ""), df[mcq_column_name].apply(lambda x: x.option_c2 if x else "")], ignore_index=True)
    result_df['option_d'] = pd.concat([df[mcq_column_name].apply(lambda x: x.option_d1 if x else ""), df[mcq_column_name].apply(lambda x: x.option_d2 if x else "")], ignore_index=True)
    result_df['correct_option'] = pd.concat([df[mcq_column_name].apply(lambda x: x.correct_option1 if x else ""),df[mcq_column_name].apply(lambda x: x.correct_option2 if x else "")],ignore_index=True)
    
    result_df.to_csv(export_filename, index=False)

def extract_json(text):
    start = text.find("{")
    end = text.find("}")
    text = text[start:end+1]
    return json.dumps(ast.literal_eval(text), ensure_ascii=False)

def generate_mcq_hf(content, model_name,tokenizer, temperature):
    prompt = f"""
        À partir du contenu éducatif suivant, générez exactement deux questions à choix multiple avec quatre options de réponse chacune (a, b, c, d), dont une seule est correcte.

        OBJECTIFS :
        - Les questions doivent évaluer la compréhension des idées principales.
        - Les distracteurs doivent être plausibles mais incorrects.
        - Les options doivent être courtes.
        - Les deux questions doivent être fournies dans un seul et unique objet JSON.
        - Aucun texte hors JSON n’est autorisé.

        CONTRAINTES STRICTES DE SORTIE :
        1. La sortie doit être STRICTEMENT un unique objet JSON valide.
        2. Interdiction ABSOLUE d’ajouter :
        - des blocs ```json
        - plusieurs objets JSON
        - du texte avant ou après le JSON
        - des explications ou commentaires
        3. Les champs "correct_option1" et "correct_option2" doivent contenir EXACTEMENT une lettre minuscule parmi : "a", "b", "c", "d".
        4. il faut utiliser des doubles quotes : "..." et NON '...'
        5. Le JSON doit contenir EXACTEMENT les 12 champs suivants :

        {{
            "question1": "...",
            "option_a1": "...",
            "option_b1": "...",
            "option_c1": "...",
            "option_d1": "...",
            "correct_option1": "a",
            "question2": "...",
            "option_a2": "...",
            "option_b2": "...",
            "option_c2": "...",
            "option_d2": "...",
            "correct_option2": "c"
        }}

        CONTENU ÉDUCATIF :
            {content}

            INSTRUCTION FINALE :
            Répondez UNIQUEMENT avec un unique objet JSON valide, sans aucun texte en dehors.
        """

    
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        device_map="cuda",
        dtype="bfloat16"
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    response = pipe(
        messages,
        max_new_tokens=2048,
        temperature=temperature,
        top_p=1.0,
        do_sample=True,
        return_full_text=False
    )

    
    return extract_json(response[0]['generated_text'])


def get_checkpoint(opt=""):
    try:
        with open(f"../data/checkpoints/start{opt}", "r") as start:
            start = start.readline()
            df_in_construction = pd.read_csv(f"../data/checkpoints/df_in_construction{opt}.csv")
    except FileNotFoundError:
        df_in_construction = pd.DataFrame()
        start = 0
    return int(start), df_in_construction

def save_checkpoint(start, df_in_construction,opt=""):
    with open(f"../data/checkpoints/start{opt}", "w") as fic:
        fic.write(str(start))
    df_in_construction.to_csv(f"../data/checkpoints/df_in_construction{opt}.csv", index=False)

def for_a_model(df_test,model_name,save_name,opt=""):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True
    )
    start, df_in_construction = get_checkpoint(opt)
    pas = 400
    for idx in range(start, len(df_test)):
        print(idx)
        content = df_test.loc[idx, "content_raw"]
        
        while True:
            try:
                generated = generate_mcq_hf(content, model_name, tokenizer, temperature=0.1)
                break
            except SyntaxError:
                print("SyntaxError détectée, relance...")

        df_in_construction.loc[idx, f"generated_{save_name}"] = generated
        if idx % pas == 0:
            save_checkpoint(idx, df_in_construction,opt)

    df_test[save_name] = df_in_construction[f'generated_{save_name}'].apply(validate_mcq)
    flatten_and_export_mcq(df_test, f'../data/base_models/instruct/{save_name}.csv', save_name)
    # Clean for other model
    os.remove(f"../data/checkpoints/df_in_construction{opt}.csv")
    os.remove(f"../data/checkpoints/start{opt}")


def main():
    load_dotenv()                  
    HF_TOKEN = os.getenv("HF_TOKEN")  
    login(token=HF_TOKEN)
    df = pd.read_csv("../data/lisa_sheets.csv")
    file_path = "../data/train_test_split/test_folders.json"
    with open(file_path, "r", encoding="utf-8") as file:
        test_folders = json.load(file)

    df_test = df[df.folder.isin(test_folders)].reset_index(drop=True)
    print("Number of lisa sheets :", len(df_test))

    print("Start EuroLLM")
    model_name = "utter-project/EuroLLM-9B-Instruct"
    save_name = "eurollm_9b"
    for_a_model(df_test,model_name,save_name)

    print("Start MedGemma")
    model_name = "google/medgemma-27b-it"
    save_name = "medGemma_27b"
    for_a_model(df_test,model_name,save_name,opt="_")


if __name__ == "__main__":
    main()

