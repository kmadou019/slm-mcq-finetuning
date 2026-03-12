import json
import re
from time import sleep

import pandas as pd
from openai import OpenAI

from eval.llm_evaluation import generate_prompt_for_question, call_openai_api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_rang(content_raw: str) -> str:
    """Extract LISA Rang (A or B) from raw content_raw field. Defaults to 'B' (stricter)."""
    match = re.search(r'\|Rang=([A-Za-z])', str(content_raw))
    return match.group(1).upper() if match else "B"


def _parse_distractor_detail(text: str) -> dict | None:
    """Parse GPT-4o JSON response into a structured detail dict.

    Returns:
        {
            "scores":  [int, int, int],       # one per distractor
            "justifs": [str, str, str],
            "avg":     float,
        }
        or None on parse failure.
    """
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text).strip()
    try:
        data = json.loads(text)
        scores  = [int(item["score"])        for item in data]
        justifs = [item.get("justif", "")    for item in data]
        return {"scores": scores, "justifs": justifs, "avg": sum(scores) / len(scores)}
    except Exception:
        return None


def _apply_threshold(detail: dict, rang: str = "B") -> bool:
    """Majority vote + no-catastrophic-distractor rule, calibrated by LISA Rang.

    Rang A (recall): threshold >= 3  — prevents blind guessing without over-engineering
    Rang B (reasoning): threshold >= 4  — ensures distractors probe real misconceptions

    Additional rule: min(scores) >= 2 to reject any manifestly implausible distractor.
    """
    scores    = detail["scores"]
    threshold = 3 if rang.upper() == "A" else 4
    majority  = sum(s >= threshold for s in scores) >= 2
    no_catastrophic = min(scores) >= 2
    return majority and no_catastrophic


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

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
                                      max_completion_tokens: int,
                                      context_col: str | None = None,
                                      lisa_sheet_col: str | None = None):
    """Evaluate distractor quality for each MCQ row.

    Stores two new columns:
    - ``distractor_quality_col``          : bool — pass/fail per rang-based threshold
    - ``distractor_quality_col + '_detail'``: JSON string with per-distractor scores and justifications
    """
    client = OpenAI(api_key=api_key)

    def distractor_quality_applicable(row):
        sleep(0.4)
        user_prompt = generate_prompt_for_question(
            row,
            question_col=question_col,
            correct_option=correct_option_col,
            option_a_col=option_a_col,
            option_b_col=option_b_col,
            option_c_col=option_c_col,
            option_d_col=option_d_col,
            include_options=True,
            context_col=context_col,
        )
        response = call_openai_api(client, system_prompt, user_prompt,
                                   temp=temp, max_completion_tokens=4000)
        detail = _parse_distractor_detail(response) if response else None
        if detail is None:
            return None
        rang = _extract_rang(row[lisa_sheet_col]) if lisa_sheet_col and lisa_sheet_col in row.index else "B"
        detail["pass"] = _apply_threshold(detail, rang=rang)
        detail["rang"] = rang
        print(f"Distractor scores: {detail['scores']}  avg={detail['avg']:.2f}  rang={rang}  pass={detail['pass']}")
        return json.dumps(detail, ensure_ascii=False)

    raw = df.apply(distractor_quality_applicable, axis=1)
    df[distractor_quality_col + '_detail'] = raw
    df[distractor_quality_col] = raw.apply(
        lambda x: json.loads(x)["pass"] if x else False
    )
    return df
