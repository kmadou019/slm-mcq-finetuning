import pandas as pd
import torch.nn.functional as F

from eval.utils import load_model, generate_embedding


def calculate_relevance_for_df(df: pd.DataFrame,
                               relevance_col: str,
                               question_col: str,
                               lisa_sheet_col: str,
                               model_name="almanach/moderncamembert-base"):
    model, tokenizer, device = load_model(model_name)

    def relevance_score(row):
        q_emb = generate_embedding(row[question_col], model, tokenizer, device)
        l_emb = generate_embedding(row[lisa_sheet_col], model, tokenizer, device)
        similarity = F.cosine_similarity(q_emb, l_emb, dim=0).item()
        return similarity

    df[relevance_col] = df.apply(relevance_score, axis=1)
    return df

