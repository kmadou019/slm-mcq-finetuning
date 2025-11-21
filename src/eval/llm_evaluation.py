from tqdm import tqdm

tqdm.pandas()

from pydantic import BaseModel


def call_openai_api(client, system_prompt, user_prompt, temp=0.5, max_completion_tokens = 1):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=temp,
            max_completion_tokens=max_completion_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        print("Token usage:", response.usage.total_tokens)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

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