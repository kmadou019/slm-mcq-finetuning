#!/usr/bin/env python3

# # Prerequisite

import os
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
import ollama
from openai import OpenAI



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
        


def flatten(df: DataFrame, mcq_column_name: str):
    # Créer les listes de données en intercalant question1/question2
    ids = []
    questions = []
    option_as = []
    option_bs = []
    option_cs = []
    option_ds = []
    correct_options = []
    
    for idx, row in df.iterrows():
        mcq = row[mcq_column_name]
        
        # Ajouter question1 et ses options
        ids.append(row["id"])
        questions.append(mcq.question1 if mcq else "")
        option_as.append(mcq.option_a1 if mcq else "")
        option_bs.append(mcq.option_b1 if mcq else "")
        option_cs.append(mcq.option_c1 if mcq else "")
        option_ds.append(mcq.option_d1 if mcq else "")
        correct_options.append(mcq.correct_option1 if mcq else "")
        
        # Ajouter question2 et ses options
        ids.append(f"{row['id']}-")
        questions.append(mcq.question2 if mcq else "")
        option_as.append(mcq.option_a2 if mcq else "")
        option_bs.append(mcq.option_b2 if mcq else "")
        option_cs.append(mcq.option_c2 if mcq else "")
        option_ds.append(mcq.option_d2 if mcq else "")
        correct_options.append(mcq.correct_option2 if mcq else "")
    
    # Créer le DataFrame avec les listes intercalées
    result_df = pd.DataFrame({
        "id": ids,
        "question": questions,
        "option_a": option_as,
        "option_b": option_bs,
        "option_c": option_cs,
        "option_d": option_ds,
        "correct_option": correct_options
    })
    
    return result_df

def extract_json(text):
    start = text.find("{")
    end = text.find("}")
    text = text[start:end+1]
    return json.dumps(ast.literal_eval(text), ensure_ascii=False)


def generate_mcq(content, model_name, temperature):
    prompt = f"""
        À partir du contenu éducatif suivant, générez deux questions à choix multiple avec quatre options de réponse dont une seule est correcte.
        La question doit évaluer la compréhension des idées principales, et les options doivent être claires, informatives et pertinentes.
        Assurez-vous que les distracteurs (options incorrectes) suivent une interprétation logique mais incorrecte, basée sur des idées reçues ou des incompréhensions courantes du sujet.
        Les options de réponse doivent être aussi courtes que possible.

        IMPORTANT — FORMAT ABSOLU POUR LES CHAMPS 'correct_option1' ET 'correct_option2' :
        - Ces champs doivent contenir exactement **une seule lettre minuscule** parmi : a, b, c ou d.
        - **Exemples valides** : "a", "b", "c", "d".
        - **Interdits** : "a)", "A", "a.", "a )", "le texte de la réponse correcte", 1, true, etc.
        - La sortie JSON doit conserver ces champs comme chaînes (`"correct_option1": "a"`).

        Fournissez la sortie strictement au format JSON correspondant au schéma demandé (ne pas produire de texte hors-du-JSON).
        **Contenu éducatif :**
        {content}
    """
    
    generate_params = {
        'model': model_name,
        'options': {'temperature': temperature, 'num_ctx': 8192, 'top_p': 1}, 
        'prompt': prompt,
        'format': MCQQuestion.model_json_schema()
    }
    
    # Get a response
    response = generate(**generate_params)
    return response['response']

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

def get_checkpoint():
    try:
        with open("../data/checkpoints/start", "r") as start:
            start = start.readline()
            df_in_construction = pd.read_csv("../data/checkpoints/df_in_construction_.csv")
    except FileNotFoundError:
        df_in_construction = pd.DataFrame()
        start = 0
    return int(start), df_in_construction

def save_checkpoint(start, df_in_construction):
    with open("../data/checkpoints/start_", "w") as fic:
        fic.write(str(start))
    df_in_construction.to_csv("../data/checkpoints/df_in_construction_.csv", index=False)

def for_a_model(df_test, model_name, save_name, use_ollama=False):
    if not use_ollama:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True
        )
    else:
        tokenizer = None
    
    start, df_in_construction = get_checkpoint()
    pas = 400
    
    for idx in range(start, len(df_test)):
        content = df_test.loc[idx, "content_raw"]
        nb_try = 0
        while True:
            try:
                generated = (
                    generate_mcq_hf(content, model_name, tokenizer, temperature=0.1)
                    if not use_ollama
                    else generate_mcq(content, model_name, temperature=0.1)
                )
                df_in_construction.loc[idx, f"generated_{save_name}"] = generated
                break
            except SyntaxError:
                print("SyntaxError détectée, relance...")
                nb_try += 1
                if nb_try == 5:
                    print("Nombre d'essai depassé, passage au Lisa Sheet suivant")
                    break
        
        if idx % pas == 0:
            save_checkpoint(idx, df_in_construction)
    
    df_test[save_name] = df_in_construction[f'generated_{save_name}'].apply(validate_mcq)
    df = flatten(df_test, save_name)
    
    # Clean for other model
    os.remove("../data/checkpoints/df_in_construction_.csv")
    os.remove("../data/checkpoints/start_")

    return df

def create_mcq_text(mcq_dict):
    return (
        f"Question: {mcq_dict['question']}\n"
        f"a) {mcq_dict['option_a']}\n"
        f"b) {mcq_dict['option_b']}\n"
        f"c) {mcq_dict['option_c']}\n"
        f"d) {mcq_dict['option_d']}"
    )

def llama_answer_qcm(mcq_text,system_prompt):
    user_prompt = f"""Répond STRICTEMENT à ce QCM :
        {mcq_text}
        CONTRAINTE ABSOLUE :
        - Ta sortie doit être UNIQUEMENT la lettre de la bonne réponse (A, B, C, D, etc.).
        - AUCUN autre texte, aucune explication, aucun point, aucun saut de ligne, aucun espace.
        - Ne préfixe pas la réponse, n’ajoute rien avant ou après.
        - Répond par une seule lettre et rien d’autre.

        FORMAT DE SORTIE OBLIGATOIRE :
        <lettre>
    """
    response = ollama.generate(
        model="llama3.1:70b",
        prompt=user_prompt,
        system=system_prompt)

    return response["response"][0]

def call_openai_api(client, system_prompt, mcq_text, temp=0.5, max_completion_tokens=1):
    user_prompt = f"""Répond STRICTEMENT à ce QCM :
        {mcq_text}
        CONTRAINTE ABSOLUE :
        - Ta sortie doit être UNIQUEMENT la lettre de la bonne réponse (A, B, C, D, etc.).
        - AUCUN autre texte, aucune explication, aucun point, aucun saut de ligne, aucun espace.
        - Ne préfixe pas la réponse, n’ajoute rien avant ou après.
        - Répond par une seule lettre et rien d’autre.

        FORMAT DE SORTIE OBLIGATOIRE :
        <lettre>
         """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=temp,
            max_tokens=max_completion_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

load_dotenv()                  
HF_TOKEN = os.getenv("HF_TOKEN")  
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
login(token=HF_TOKEN)

df = pd.read_csv("../data/lisa_sheets.csv")

file_path = "../data/train_test_split/test_folders.json"

with open(file_path, "r", encoding="utf-8") as file:
    test_folders = json.load(file)

df_test = df[df.folder.isin(test_folders)].reset_index(drop=True)
print("Number of lisa sheets :", len(df_test))

# # How many output are incorrect

def correct_output(df,save_name,file):
    initial_len = len(df)
    df = df[df["correct_option"].astype(str).str.lower().isin(list("abcd"))]
    
    incorrect_output = initial_len - len(df)
    print(initial_len-incorrect_output)
    print(f"Incorrect output for {save_name}: {round((incorrect_output/initial_len)*100,2)}%",file=file)
    return df

# ## Correctness

system_prompt = "Tu es un expert dans le domaine médical"
client = OpenAI(api_key=OPENAI_KEY)

def correctness(df,save_name,file):
    initial_len = len(df)
    indices_to_drop = []
    for idx, row in df.iterrows():
        mcq_text = create_mcq_text(row)
        correct_option = row["correct_option"]
        correct_reeval = call_openai_api(client=client,system_prompt=system_prompt,mcq_text=mcq_text)
        if  not(isinstance(correct_option, str)):
            continue 
        if correct_option.lower() != correct_reeval.lower():
            indices_to_drop.append(idx)

    df = df.drop(indices_to_drop).reset_index(drop=True)
    df.to_csv("../data/correct_mcqs_dataset/"+ save_name + ".csv")
    print(f"Number of correct MCQs for {save_name} {len(df)} / {initial_len}",file=file)

# # MCQs Generation

models = {
    #"qwen3_8b_pdapt_slerp": "PARTAGES-dev/Qwen3-8B-PDAPT-SLERP",
    "qwen3_4b_pdapt_slerp": "PARTAGES-dev/Qwen3-4B-PDAPT-SLERP",
    "qwen3_1.7b_pdapt_slerp": "PARTAGES-dev/Qwen3-1.7B-PDAPT-SLERP",
    "qwen3_0.6b_pdapt_slerp": "PARTAGES-dev/Qwen3-0.6B-PDAPT-SLERP",
    #"llama3_1_8b": "meta-llama/Llama-3.1-8B-Instruct",
    #"gemma2_9b": "google/gemma-2-9b-it",
    #"medGemma_4b": "google/medgemma-4b-it",
    #"medGemma_27b": "google/medgemma-27b-it",
    #"openbiollm_8b": "hf.co/mradermacher/Llama3-Instruct-OpenBioLLM-8B-merged-i1-GGUF:latest",
    #"qwen3_0.6b": "Qwen/Qwen3-0.6B",
    #"mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3",
    #"eurollm_9b": "utter-project/EuroLLM-9B-Instruct",
    #"apertus_8b": "swiss-ai/Apertus-8B-Instruct-2509",
}

with open("correctness.output", mode="a") as f:
    for save_name,model_name in models.items():
        df = for_a_model(df_test,model_name,save_name)
        df = correct_output(df,save_name,f)
        correctness(df,save_name,f)


