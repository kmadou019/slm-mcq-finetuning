import json
from time import sleep
import pandas as pd
from openai import OpenAI
import re

from eval.llm_evaluation import generate_prompt_for_question, call_openai_api, call_llama3 #, create_client

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
    
    return sum(s >= 4 for s in scores) >= 2


def compute_distractor_quality_for_df(df: pd.DataFrame,
                                      api_key: str,
                                      question_col: str,
                                      correct_option_col: str,
                                      option_a_col: str,
                                      option_b_col: str,
                                      option_c_col: str,
                                      option_d_col: str,
                                      distractor_quality_col: str,
                                      system_prompt: str,
                                      temp: float,
                                      max_completion_tokens: int):

    #client = create_client(api_key)
    client = OpenAI(api_key = api_key)
    
    def distractor_quality_applicable(row):
        sleep(0.4) #Uncomment for gpt for api rate limit
        user_prompt = generate_prompt_for_question(row,
                                                   question_col=question_col,
                                                   correct_option=correct_option_col,
                                                   option_a_col=option_a_col,
                                                   option_b_col=option_b_col,
                                                   option_c_col=option_c_col,
                                                   option_d_col=option_d_col,
                                                   include_options=True)

        response = call_openai_api(client, system_prompt, user_prompt, temp=temp, max_completion_tokens=4000)
        score = extraire_moyenne_scores(response)
        print("Score:", score)
        return score
    
    df[distractor_quality_col] = df.apply(distractor_quality_applicable, axis=1)
    return df