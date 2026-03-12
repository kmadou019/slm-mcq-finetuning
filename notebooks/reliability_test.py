"""
reliability_test.py
--------------------
Validates the reliability of the GPT-4o distractor quality metric.

Two tests:
  4.1 — Intra-rater  : submit the same N MCQs twice to GPT-4o, measure score variance.
  4.2 — Inter-rater  : export a CSV template for manual annotation, then compute
                       Cohen's κ once the file is filled in.

Usage:
    cd notebooks
    # Run both tests (generates MCQs from first N LISA rows)
    python reliability_test.py --n 20

    # Once you have filled reliability_manual_annotations.csv:
    python reliability_test.py --kappa
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'page', 'backend'))

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'src', 'page', 'backend', '.env'))

from api.utils.generate_mcq_GPU import generate_mcq, _HF_PROMPT_TEMPLATE, validate_mcq, shuffle_distractors
from eval.distractors_quality import compute_distractor_quality_for_df

OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
LISA_CSV    = os.path.join(os.path.dirname(__file__), '..', 'src', 'page', 'backend', 'data', 'lisa_sheets.csv')
PROMPTS_JSON = os.path.join(os.path.dirname(__file__), '..', 'src', 'page', 'backend', 'eval', 'prompts.json')
OUT_DIR     = Path(os.path.dirname(__file__)) / '..' / 'docs'
MODEL       = "magistral:latest"   # use the faster model for reliability testing


# ---------------------------------------------------------------------------
# 4.1 — Intra-rater test
# ---------------------------------------------------------------------------

def run_intra_rater(n: int):
    """Submit the same N MCQs to GPT-4o twice, compare scores."""
    print(f"\n=== 4.1 Intra-rater test (n={n}) ===")

    with open(PROMPTS_JSON) as f:
        prompts = json.load(f)
    system_prompt = prompts['distractors_quality_prompt']

    df_lisa = pd.read_csv(LISA_CSV, nrows=n)

    # Generate MCQs once
    rows = []
    for _, lisa_row in df_lisa.iterrows():
        prompt = _HF_PROMPT_TEMPLATE.format(content=lisa_row['content_raw'])
        try:
            mcq = validate_mcq(generate_mcq(prompt, MODEL))
            if mcq:
                mcq = shuffle_distractors(mcq)
                rows.append({
                    'lisa_id':      lisa_row['id'],
                    'content_raw':  lisa_row['content_raw'],
                    'question':     mcq.question,
                    'option_a':     mcq.option_a, 'option_a_comment': mcq.option_a_comment,
                    'option_b':     mcq.option_b, 'option_b_comment': mcq.option_b_comment,
                    'option_c':     mcq.option_c, 'option_c_comment': mcq.option_c_comment,
                    'option_d':     mcq.option_d, 'option_d_comment': mcq.option_d_comment,
                    'correct_option': mcq.correct_option,
                })
        except Exception as e:
            print(f"  ✗ {lisa_row['id']} — {e}")

    if not rows:
        print("No MCQs generated, aborting.")
        return

    df = pd.DataFrame(rows)
    print(f"Generated {len(df)} MCQs. Running GPT-4o evaluation twice...")

    # Pass 1
    df1 = compute_distractor_quality_for_df(
        df.copy(), api_key=OPENAI_KEY,
        question_col='question', correct_option_col='correct_option',
        option_a_col='option_a', option_b_col='option_b',
        option_c_col='option_c', option_d_col='option_d',
        distractor_quality_col='dq',
        system_prompt=system_prompt, temp=0.1, max_completion_tokens=4000,
        context_col='content_raw', lisa_sheet_col='content_raw',
    )

    # Pass 2
    df2 = compute_distractor_quality_for_df(
        df.copy(), api_key=OPENAI_KEY,
        question_col='question', correct_option_col='correct_option',
        option_a_col='option_a', option_b_col='option_b',
        option_c_col='option_c', option_d_col='option_d',
        distractor_quality_col='dq',
        system_prompt=system_prompt, temp=0.1, max_completion_tokens=4000,
        context_col='content_raw', lisa_sheet_col='content_raw',
    )

    # Compare
    records = []
    for idx in range(len(df)):
        try:
            d1 = json.loads(df1.iloc[idx]['dq_detail'])
            d2 = json.loads(df2.iloc[idx]['dq_detail'])
            s1, s2 = d1['scores'], d2['scores']
            diffs = [abs(a - b) for a, b in zip(s1, s2)]
            records.append({
                'lisa_id':      df.iloc[idx]['lisa_id'],
                'scores_pass1': s1,
                'scores_pass2': s2,
                'avg_pass1':    d1['avg'],
                'avg_pass2':    d2['avg'],
                'pass_pass1':   d1['pass'],
                'pass_pass2':   d2['pass'],
                'per_distractor_abs_diff': diffs,
                'mean_abs_diff': np.mean(diffs),
            })
        except Exception as e:
            print(f"  Parse error at index {idx}: {e}")

    result_df = pd.DataFrame(records)
    out_path = OUT_DIR / 'reliability_intra_rater.csv'
    result_df.to_csv(out_path, index=False)

    # Summary stats
    all_diffs = result_df['mean_abs_diff'].dropna()
    agree     = result_df['pass_pass1'] == result_df['pass_pass2']
    print(f"\n--- Intra-rater summary ---")
    print(f"  Mean absolute score diff (per distractor): {all_diffs.mean():.3f}")
    print(f"  Std dev:                                   {all_diffs.std():.3f}")
    print(f"  Max diff:                                  {all_diffs.max():.3f}")
    print(f"  Pass/fail agreement rate:                  {agree.mean()*100:.1f}%")
    print(f"\n  Interpretation:")
    print(f"    Mean diff < 0.5 → stable  ✓" if all_diffs.mean() < 0.5 else
          f"    Mean diff ≥ 0.5 → unstable, consider averaging multiple passes")
    print(f"\n  Full results saved to: {out_path}")


# ---------------------------------------------------------------------------
# 4.2 — Inter-rater: export annotation template
# ---------------------------------------------------------------------------

def export_annotation_template(n: int):
    """Generate MCQs and export a CSV for manual annotation."""
    print(f"\n=== 4.2 Inter-rater — exporting annotation template (n={n}) ===")

    with open(PROMPTS_JSON) as f:
        prompts = json.load(f)
    system_prompt = prompts['distractors_quality_prompt']

    df_lisa = pd.read_csv(LISA_CSV, nrows=n)
    rows = []
    for _, lisa_row in df_lisa.iterrows():
        prompt = _HF_PROMPT_TEMPLATE.format(content=lisa_row['content_raw'])
        try:
            mcq = validate_mcq(generate_mcq(prompt, MODEL))
            if mcq:
                mcq = shuffle_distractors(mcq)
                rows.append({
                    'lisa_id':       lisa_row['id'],
                    'content_raw':   lisa_row['content_raw'],
                    'question':      mcq.question,
                    'option_a':      mcq.option_a,
                    'option_b':      mcq.option_b,
                    'option_c':      mcq.option_c,
                    'option_d':      mcq.option_d,
                    'correct_option': mcq.correct_option,
                    # Columns to fill manually (1-5 per distractor)
                    'manual_score_b': '',
                    'manual_score_c': '',
                    'manual_score_d': '',
                    'manual_notes':   '',
                })
        except Exception as e:
            print(f"  ✗ {lisa_row['id']} — {e}")

    df = pd.DataFrame(rows)

    # Also run GPT-4o for comparison
    print("Running GPT-4o evaluation for comparison column...")
    eval_df = compute_distractor_quality_for_df(
        df[['lisa_id', 'content_raw', 'question', 'option_a', 'option_b',
            'option_c', 'option_d', 'correct_option']].copy(),
        api_key=OPENAI_KEY,
        question_col='question', correct_option_col='correct_option',
        option_a_col='option_a', option_b_col='option_b',
        option_c_col='option_c', option_d_col='option_d',
        distractor_quality_col='dq',
        system_prompt=system_prompt, temp=0.1, max_completion_tokens=4000,
        context_col='content_raw', lisa_sheet_col='content_raw',
    )

    df['gpt4o_scores']   = eval_df['dq_detail'].apply(
        lambda x: json.loads(x)['scores'] if x else None
    )
    df['gpt4o_avg']      = eval_df['dq_detail'].apply(
        lambda x: json.loads(x)['avg'] if x else None
    )
    df['gpt4o_pass']     = eval_df['dq']
    df['gpt4o_justifs']  = eval_df['dq_detail'].apply(
        lambda x: json.loads(x)['justifs'] if x else None
    )

    out_path = OUT_DIR / 'reliability_manual_annotations.csv'
    df.to_csv(out_path, index=False)
    print(f"\n  Template saved to: {out_path}")
    print(f"  → Fill in 'manual_score_b', 'manual_score_c', 'manual_score_d' (scale 1-5)")
    print(f"  → Then run: python reliability_test.py --kappa")


# ---------------------------------------------------------------------------
# 4.2 — Inter-rater: compute Cohen's κ from filled template
# ---------------------------------------------------------------------------

def compute_kappa():
    """Compute Cohen's κ between manual annotations and GPT-4o scores."""
    from sklearn.metrics import cohen_kappa_score

    annotation_path = OUT_DIR / 'reliability_manual_annotations.csv'
    if not annotation_path.exists():
        print(f"File not found: {annotation_path}")
        print("Run without --kappa first to generate the template.")
        return

    df = pd.read_csv(annotation_path)
    required = ['manual_score_b', 'manual_score_c', 'manual_score_d', 'gpt4o_scores']
    if df[required[:3]].isnull().all().any():
        print("Manual annotation columns are empty. Please fill in the CSV first.")
        return

    manual_scores, gpt_scores = [], []
    for _, row in df.iterrows():
        try:
            m = [int(row['manual_score_b']), int(row['manual_score_c']), int(row['manual_score_d'])]
            g = json.loads(row['gpt4o_scores']) if isinstance(row['gpt4o_scores'], str) else row['gpt4o_scores']
            manual_scores.extend(m)
            gpt_scores.extend(g)
        except Exception:
            continue

    if not manual_scores:
        print("Could not parse scores. Check CSV format.")
        return

    kappa = cohen_kappa_score(manual_scores, gpt_scores, weights='linear')
    mae   = np.mean([abs(m - g) for m, g in zip(manual_scores, gpt_scores)])

    print(f"\n--- Inter-rater results ---")
    print(f"  N distractor comparisons : {len(manual_scores)}")
    print(f"  Linear Cohen's κ         : {kappa:.3f}")
    print(f"  Mean absolute error      : {mae:.3f}")
    print()
    if kappa < 0.2:
        print("  ✗ κ < 0.20 — Agreement near chance. Rubric needs redesign.")
    elif kappa < 0.4:
        print("  △ κ 0.20–0.40 — Fair agreement. Rubric needs clarification.")
    elif kappa < 0.6:
        print("  ~ κ 0.41–0.60 — Moderate agreement. Acceptable for exploration.")
    elif kappa < 0.8:
        print("  ✓ κ 0.61–0.80 — Good agreement. Publishable.")
    else:
        print("  ✓✓ κ > 0.80 — Excellent agreement.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',     type=int, default=20, help='Number of MCQs to generate')
    parser.add_argument('--kappa', action='store_true',  help='Compute κ from filled annotation CSV')
    args = parser.parse_args()

    if args.kappa:
        compute_kappa()
    else:
        run_intra_rater(args.n)
        export_annotation_template(args.n)
