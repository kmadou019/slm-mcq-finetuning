#!/usr/bin/env python3
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, pipeline
import re
import pandas as pd

dataset = load_dataset("openlifescienceai/medmcqa")

def clean(text):
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

def generate_prompt_for_question(row,
                                 question_col='question',
                                 option_a_col = 'opa',
                                 option_b_col = 'opb',
                                 option_c_col = 'opc',
                                 option_d_col = 'opd',
                                 correct_option = 'cop',
                                 include_options=True,
                                 include_correct_option = False,
                                 context_col=None):
    question_text = row[question_col]
    options = f"a) {row[option_a_col]}\nb) {row[option_b_col]}\nc) {row[option_c_col]}\nd) {row[option_d_col]}"
    correct_option = row[correct_option]
    
    user_prompt_delimiter = "-----\n"
    user_prompt_question = f"Question:\n{question_text}\n"
    user_prompt_options = f"Options:\n{options}\n"

    user_prompt = user_prompt_delimiter + user_prompt_question
    if include_options:
        user_prompt += user_prompt_options
    if include_correct_option:
        correct_option = f"Correct option: {correct_option}\n"
        user_prompt += correct_option

    user_prompt += user_prompt_delimiter
    
    if context_col is not None:
        mcq_context = f"""Context:\n-----\n{row[context_col]}\n-----\n"""
        user_prompt = mcq_context + user_prompt

    return user_prompt

def answer_mcq_hf(mcq, model_name,tokenizer, temperature):
    prompt = f"""
       You are tasked with answering multiple-choice questions, containing 4 different answer options - a, b, c and d.
       You are given some context to help you answer the question.
       Provide just a single letter corresponding to the correct option as the response.
       For example: "a", "b", "c", or "d"
    """

    
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        device_map="cuda",
        dtype="bfloat16"
    )
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": mcq}
    ]
    
    response = pipe(
        messages,
        max_new_tokens=2048,
        temperature=temperature,
        top_p=1.0,
        do_sample=True,
        return_full_text=False
    )
    return clean(response[0]['generated_text'])[0]

def answer_mcq(mcq, model_name,tokenizer, temperature):
    ...

def for_a_model(dataset, model_name, save_name, use_ollama=False):
    if not use_ollama:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True
        )
    else:
        tokenizer = None
    
    result = pd.DataFrame(columns=["mcq", "cop", "llm_cop"])
    conv = {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3
    }
    
    for idx in range(len(dataset)):
        mcq = generate_prompt_for_question(dataset.loc[idx])
        nb_try = 0
        
        while True:
            try:
                generated = (
                    answer_mcq_hf(mcq, model_name, tokenizer, temperature=0.1)
                    if not use_ollama
                    else answer_mcq(mcq, model_name, temperature=0.1)
                )
                generated = conv[generated]
                break
            except (SyntaxError, KeyError):
                print("SyntaxError ou KeyError détectée, relance...")
                nb_try += 1
                if nb_try == 5:
                    print("Nombre d'essai dépassé, passage au Lisa Sheet suivant")
                    generated = None
                    break
        
        if generated is not None:
            result.loc[idx] = [mcq, dataset.loc[idx]["cop"], generated]
    
    result.to_csv(f"../data/answerability/{save_name}.csv", index=False)
    return result

all_dataset = concatenate_datasets([dataset["train"],dataset["test"],dataset["validation"]])
df = all_dataset.to_pandas()
df_grouped = df.groupby("subject_name")

dataset_answerability = pd.concat(
    [group.sample(n=min(200, len(group))) for name, group in df_grouped if name != "Unknown"],
    ignore_index=True
)
print(len(dataset_answerability))


models = [  
      #Qwen
          #("Qwen/Qwen3-0.6B","qwen3_0.6b"),
          ("Qwen/Qwen3-4B-Instruct-2507","qwen3_4b"),
          ("mistralai/Mistral-7B-Instruct-v0.3","mistral_7b"),
          ("utter-project/EuroLLM-9B-Instruct","eurollm_9b"),
          ("swiss-ai/Apertus-8B-Instruct-2509","apertus_8B"),
          ("Qwen/Qwen3-1.7B","qwen3_1.7b"),
          #("Qwen/Qwen3-8B-Base","qwen3_8b"),

          #("google/gemma-2-9b-it","gemma2_9b"),
          #("google/medgemma-27b-it","medGemma_27b"),
          #("hf.co/mradermacher/Llama3-Instruct-OpenBioLLM-8B-merged-i1-GGUF:latest", "openbiollm_8b"),
          #("meta-llama/Llama-3.1-8B-Instruct","llama3_1_8b"),
          #("google/medgemma-4b-it","medGemma_4b"),
      
        ]


with open("answerability_output.txt", "a") as f:
    print("MedMCQA", file=f)
    for model_name, save_name in models:
        result = for_a_model(dataset_answerability, model_name, save_name)
        percentage = (result["cop"] == result["llm_cop"]).mean() * 100
        print(f"Answerability of {model_name}: {percentage:.2f}%", file=f)
    
    print("UNESS", file=f)
    dataset_answerability = pd.read_json("../data/mcqs_answerability.json")
    for model_name, save_name in models:
        result = for_a_model(dataset_answerability, model_name, save_name + "_UNESS")
        percentage = (result["cop"] == result["llm_cop"]).mean() * 100
        print(f"Answerability of {model_name}: {percentage:.2f}%", file=f)


