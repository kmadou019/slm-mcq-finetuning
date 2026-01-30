from tqdm import tqdm
import ollama
tqdm.pandas()
import json
import re
from pydantic import BaseModel


def extraire_moyenne_scores(text):
    """
    Extrait une liste de dictionnaires JSON d'un texte et retourne la moyenne des scores.
    Gère le cas où le texte contient ```json ... ```
    
    Args:
        text (str): Texte contenant du JSON (avec ou sans balises ```json)
    
    Returns:
        float: Moyenne des scores
    """
    # Supprimer les balises ```json et ```
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Parser le JSON
    data = json.loads(text)
    scores = [item["score"] for item in data]
    
    return sum(scores) / len(scores)


def call_openai_api(client, system_prompt, user_prompt, temp=0.5, max_completion_tokens = 1):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=temp,
            max_completion_tokens=4000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        response  = response.choices[0].message.content
        score = extraire_moyenne_scores(response)
        print("Score:", score)
        return score
    except Exception as e:
        print(f"Error occurred: {e}")
        return None
    

def call_llama3(system_prompt, user_prompt, temp=0.5):
    response = ollama.generate(
        model="llama3.1:70b",
        system=system_prompt,
        prompt=user_prompt,
        options={
            "temperature":temp
        })
    print(response["response"])
    return response["response"]

def generate_prompt_for_question(row,
                                 question_col='question',
                                 option_a_col = 'option_a',
                                 option_b_col = 'option_b',
                                 option_c_col = 'option_c',
                                 option_d_col = 'option_d',
                                 correct_option = 'correct_option',
                                 include_options=True,
                                 include_correct_option = True,
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


def process_dataframe(model_name, df):
    try:
        # df['rank'] = df.progress_apply(generate_prompt_for_question, axis=1)
        df.to_csv(f'/kaggle/working/results_of_{model_name}.csv', index=False)
    except Exception as e:
        print(f"Error occurred for model {model_name}: {e}")