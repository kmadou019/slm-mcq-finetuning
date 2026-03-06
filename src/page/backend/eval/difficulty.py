import json

import pandas as pd
from openai import OpenAI

from eval.llm_evaluation import generate_prompt_for_question, call_openai_api


def compute_difficulty_for_df(df: pd.DataFrame,
                              api_key: str,
                              question_col: str,
                              difficulty_col: str,
                              system_prompt: str,
                              temp: float,
                              max_completion_tokens: int):

    client = OpenAI(api_key = api_key)
            
    def difficulty_applicable(row):
        user_prompt = generate_prompt_for_question(row,
                                                   question_col=question_col,
                                                   include_options=True,
                                                   include_correct_option= True)
        
        return call_openai_api(client, system_prompt, user_prompt, temp=temp, max_completion_tokens=max_completion_tokens)

    df[difficulty_col] = df.apply(difficulty_applicable, axis=1)
    return df
    