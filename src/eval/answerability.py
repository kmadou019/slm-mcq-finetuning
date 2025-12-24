import pandas as pd
from openai import OpenAI

from eval.llm_evaluation import generate_prompt_for_question, call_openai_api


def compute_answerability_for_df(df: pd.DataFrame,
                                 api_key: str,
                                 question_col: str,
                                 option_a_col: str,
                                 option_b_col: str,
                                 option_c_col: str,
                                 option_d_col: str,
                                 context_col: str,
                                 model_answer_col: str,
                                 system_prompt: str,
                                 temp: float,
                                 max_completion_tokens: int):

    client = OpenAI(api_key = api_key)
            
    def answerability_applicable(row):
        user_prompt = generate_prompt_for_question(row,
                                                   question_col=question_col,
                                                   option_a_col = option_a_col,
                                                   option_b_col = option_b_col,
                                                   option_c_col = option_c_col,
                                                   option_d_col = option_d_col,
                                                   context_col=context_col)
        
        return call_openai_api(client, system_prompt, user_prompt, temp=temp, max_completion_tokens=max_completion_tokens)

    df[model_answer_col] = df.apply(answerability_applicable, axis=1)
    return df
    