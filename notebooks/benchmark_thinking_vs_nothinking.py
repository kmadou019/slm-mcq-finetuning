"""
benchmark_thinking_vs_nothinking.py
-------------------------------------
Compares thinking vs non-thinking model variants across 6 pairs on 1000 LISA rows.
Evaluates all metrics and outputs a comparison HTML report.

Pairs:
  1. Qwen3 14B  think (/think prefix) vs no_think (/no_think prefix) — same weights
  2. Magistral 24B  vs  Mistral Small 22B              — same family
  3. Nemotron 30B   think vs no_think flag              — same weights
  4. Phi-4 Reasoning 14B  vs  Phi-4 14B                — same base
  5. DeepSeek-R1 14B  vs  Qwen2.5 14B                  — same base weights (R1 distill)
  6. QwQ 32B  vs  Qwen2.5 32B                           — same family

Usage:
    cd notebooks
    python benchmark_thinking_vs_nothinking.py
    # → docs/benchmark_thinking_vs_nothinking.html
    # Checkpoints saved in docs/checkpoints/ — rerun resumes automatically.
"""

import sys
import os
import json
import math
import random
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'page', 'backend'))

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'src', 'page', 'backend', '.env'))

from api.utils.generate_mcq_GPU import generate_mcq_chat, _HF_PROMPT_TEMPLATE, validate_mcq, shuffle_distractors
from eval.eval_dataframe import eval_dataframe

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_KEY   = os.getenv("OPENAI_API_KEY")
N_LISA_ROWS  = 1000
SAMPLE_CARDS = 30   # number of LISA rows shown in the per-question detail section
RANDOM_SEED  = 42

LISA_CSV     = os.path.join(os.path.dirname(__file__), '..', 'src', 'page', 'backend', 'data', 'lisa_sheets.csv')
PROMPTS_JSON = os.path.join(os.path.dirname(__file__), '..', 'src', 'page', 'backend', 'eval', 'prompts.json')
OUTPUT_HTML  = os.path.join(os.path.dirname(__file__), '..', 'docs', 'benchmark_thinking_vs_nothinking.html')
CKPT_DIR     = Path(os.path.dirname(__file__)) / '..' / 'docs' / 'checkpoints'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Each side: name, model tag, prompt_prefix (empty string = no prefix)
PAIRS = [
    {
        "label":         "Qwen3 14B",
        "color_think":   "#8b5cf6",
        "color_nothink": "#6366f1",
        "think":   {"name": "Qwen3-14B (think)",    "model": "qwen3:14b", "think_flag": True},
        "nothink": {"name": "Qwen3-14B (no_think)", "model": "qwen3:14b", "think_flag": False},
    },
    {
        "label":         "Mistral ~24B",
        "color_think":   "#0ea5e9",
        "color_nothink": "#38bdf8",
        "think":   {"name": "Magistral 24B",     "model": "magistral:latest",     "think_flag": True},
        "nothink": {"name": "Mistral Small 24B", "model": "mistral-small3.1:24b", "think_flag": False},
    },
    {
        "label":         "Nemotron 30B",
        "color_think":   "#f59e0b",
        "color_nothink": None,
        # Nemotron always thinks via internal <think> tags — Ollama think flag not supported.
        # No non-thinking equivalent: shown as standalone thinking model (no pair).
        "think":   {"name": "Nemotron 30B", "model": "hf.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q8_0", "think_flag": False},
        "nothink": None,
    },
    {
        "label":         "Qwen2.5 14B / DeepSeek-R1 14B",
        "color_think":   "#ef4444",
        "color_nothink": "#f87171",
        # DeepSeek-R1 supports Ollama think flag — exposes trace via message.thinking
        "think":   {"name": "DeepSeek-R1 14B", "model": "deepseek-r1:14b", "think_flag": True},
        "nothink": {"name": "Qwen2.5 14B",     "model": "qwen2.5:14b",     "think_flag": False},
    },
    {
        "label":         "Qwen2.5 32B / QwQ 32B",
        "color_think":   "#ec4899",
        "color_nothink": "#f472b6",
        # QwQ does not support Ollama think API — thinks via <think> tags in content
        "think":   {"name": "QwQ 32B",     "model": "qwq:32b",     "think_flag": False},
        "nothink": {"name": "Qwen2.5 32B", "model": "qwen2.5:32b", "think_flag": False},
    },
    {
        "label":         "Phi-4 14B",
        "color_think":   "#10b981",
        "color_nothink": "#34d399",
        # Phi-4-Reasoning does not support Ollama think API — thinks via <think> tags in content
        # num_predict=8192: generous budget to avoid truncation before JSON output
        "think":   {"name": "Phi-4 Reasoning 14B", "model": "phi4-reasoning:14b", "think_flag": False, "num_predict": 8192},
        "nothink": {"name": "Phi-4 14B",            "model": "phi4:14b",           "think_flag": False},
    },
]

# Flat list of all model sides (used for global summary table)
ALL_SIDES = [(p["label"], side, p[side]) for p in PAIRS for side in ("think", "nothink") if p[side] is not None]

# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------

with open(PROMPTS_JSON) as f:
    prompts = json.load(f)

df_lisa = pd.read_csv(LISA_CSV, nrows=N_LISA_ROWS)
print(f"Loaded {len(df_lisa)} LISA rows")

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def ckpt_path(side_name: str) -> Path:
    safe = side_name.replace(" ", "_").replace("/", "-").replace(".", "_").replace(":", "_")
    return CKPT_DIR / f"thinking_benchmark_{safe}.csv"


def load_checkpoint(side_name: str) -> set[str]:
    """Return set of already-generated lisa_ids."""
    p = ckpt_path(side_name)
    if p.exists():
        try:
            df = pd.read_csv(p)
            done = set(df['lisa_id'].astype(str).tolist())
            print(f"  [ckpt] {side_name}: resuming — {len(done)} rows already done")
            return done
        except Exception:
            pass
    return set()


def append_checkpoint(side_name: str, row: dict):
    p = ckpt_path(side_name)
    df_row = pd.DataFrame([row])
    if p.exists():
        df_row.to_csv(p, mode='a', header=False, index=False)
    else:
        df_row.to_csv(p, index=False)


def read_checkpoint_df(side_name: str) -> pd.DataFrame | None:
    p = ckpt_path(side_name)
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return None

# ---------------------------------------------------------------------------
# Generation (with checkpointing)
# ---------------------------------------------------------------------------

def generate_for_side(pair_label: str, side_key: str, side_cfg: dict) -> list[dict]:
    name        = side_cfg["name"]
    model       = side_cfg["model"]
    think_flag  = side_cfg["think_flag"]
    num_predict = side_cfg.get("num_predict", 8192)

    print(f"\n[GEN] {name}  (model={model}, think={think_flag})")
    done_ids = load_checkpoint(name)
    rows = []

    # Load already-done rows from checkpoint
    ckpt_df = read_checkpoint_df(name)
    if ckpt_df is not None and not ckpt_df.empty:
        rows = ckpt_df.to_dict('records')

    for _, lisa_row in df_lisa.iterrows():
        lisa_id = str(lisa_row['id'])
        if lisa_id in done_ids:
            continue

        full_prompt = _HF_PROMPT_TEMPLATE.format(content=lisa_row['content_raw'])

        mcq = None
        for attempt in range(3):
            try:
                mcq_json = generate_mcq_chat(full_prompt, model, think=think_flag, num_predict=num_predict)
                mcq = validate_mcq(mcq_json)
                if mcq:
                    break
                print(f"  ↺ {lisa_id} — validation failed, retry {attempt+1}/3")
            except Exception as e:
                print(f"  ↺ {lisa_id} — {e}, retry {attempt+1}/3")

        if mcq:
            mcq = shuffle_distractors(mcq)
            row = {
                'lisa_id':     lisa_id,
                'folder':      lisa_row.get('folder', ''),
                'content_raw': lisa_row['content_raw'],
                'rang':        lisa_row.get('rang', ''),
                'pair_label':  pair_label,
                'side':        side_key,
                'model_name':  name,
                **mcq.model_dump(),
            }
            rows.append(row)
            append_checkpoint(name, row)
            print(f"  ✓ {lisa_id}")
        else:
            print(f"  ✗ {lisa_id} — failed after 3 attempts")

    return rows

# ---------------------------------------------------------------------------
# Run generation for all pairs
# ---------------------------------------------------------------------------

raw_results: dict[str, list[dict]] = {}   # key = side name

for pair in PAIRS:
    for side_key in ("think", "nothink"):
        side_cfg = pair[side_key]
        if side_cfg is None:
            continue
        side_name = side_cfg["name"]
        rows = generate_for_side(pair["label"], side_key, side_cfg)
        raw_results[side_name] = rows
        print(f"  → {len(rows)} MCQs collected for {side_name}")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

eval_dfs: dict[str, pd.DataFrame] = {}

for side_name, rows in raw_results.items():
    if not rows:
        print(f"[EVAL] {side_name}: no rows, skipping")
        continue
    print(f"\n[EVAL] {side_name} — {len(rows)} MCQs")
    df = pd.DataFrame(rows)
    df = eval_dataframe(
        df,
        openai_key=OPENAI_KEY,
        answerability_system_prompt=prompts['answerability_prompt'],
        disclosure_system_prompt=prompts['disclosure_prompt'],
        difficulty_system_prompt=prompts['difficulty_prompt'],
        distractors_quality_system_prompt=prompts['distractors_quality_prompt'],
        output_file_path=f'/tmp/eval_thinking_{side_name.replace(" ", "_").replace("/", "-")}.csv',
    )
    # Derive answerability_correct
    if 'gpt_answer' in df.columns and 'correct_option' in df.columns:
        df['answerability_correct'] = (
            df['gpt_answer'].astype(str).str.strip().str.lower() ==
            df['correct_option'].astype(str).str.strip().str.lower()
        )
    eval_dfs[side_name] = df
    print(f"  done")

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

METRICS_META = [
    ("is_question",           "Is a Question",        "Section A"),
    ("starts_with_negation",  "No Negation",          "Section A"),
    ("originality",           "Originality",          "Section A"),
    ("readability",           "Readability (FK)",     "Section A"),
    ("relevance",             "Relevance",            "Section B"),
    ("ambiguity",             "Ambiguity (info)",     "Section B"),
    ("answerability_correct", "Answerability",        "Section B"),
    ("disclosure",            "No Disclosure",        "Section B"),
    ("distractors_quality",   "Distractor Quality",   "Section B"),
]


def avg_metric(df: pd.DataFrame, metric: str):
    if metric not in df.columns:
        return None
    col = df[metric]
    if metric in ("originality", "readability", "relevance", "ambiguity"):
        return col.dropna().astype(float).mean()
    if metric == "is_question":
        return col.dropna().astype(bool).mean()
    if metric == "starts_with_negation":
        # Inverted: we want low negation → display as % non-negation
        return 1.0 - col.dropna().astype(bool).mean()
    if metric == "answerability_correct":
        return col.dropna().astype(bool).mean()
    if metric == "disclosure":
        # Inverted: lower disclosure rate is better
        return 1.0 - (col.dropna().astype(str).str.lower() == "true").mean()
    if metric == "distractors_quality":
        detail_col = "distractors_quality_detail"
        col = df[detail_col] if detail_col in df.columns else col
        scores = []
        for v in col.dropna():
            try:
                detail = json.loads(v) if isinstance(v, str) else v
                if isinstance(detail, dict):
                    scores.append(float(detail.get('avg', 0)))
                elif isinstance(detail, list):
                    scores.append(sum(d["score"] for d in detail) / len(detail))
            except Exception:
                pass
        return sum(scores) / len(scores) if scores else None
    return None


def fmt_avg(metric: str, avg) -> tuple[str, str]:
    """Return (display_text, css_class)."""
    if avg is None or (isinstance(avg, float) and math.isnan(avg)):
        return "—", "na"

    if metric in ("originality", "relevance"):
        cls = "good" if avg >= 0.5 else "bad"
        return f"{avg:.2f}", cls

    if metric == "readability":
        cls = "good" if avg >= 12 else "bad"
        return f"{avg:.1f}", cls

    if metric == "ambiguity":
        # informative only — show score but no good/bad
        return f"{avg:.2f}", "neutral"

    if metric in ("is_question", "starts_with_negation", "answerability_correct", "disclosure"):
        pct = avg * 100
        cls = "good" if pct >= 80 else ("warn" if pct >= 50 else "bad")
        return f"{pct:.0f}%", cls

    if metric == "distractors_quality":
        cls = "good" if avg >= 4 else ("warn" if avg >= 3 else "bad")
        return f"{avg:.2f}/5", cls

    return str(avg), "neutral"


def build_global_summary(pairs: list, eval_dfs: dict) -> str:
    """One summary table per pair. Pairs with nothink=None render as single-column."""
    html = ""
    for pair in pairs:
        think_name = pair["think"]["name"]
        solo       = pair["nothink"] is None
        ct         = pair["color_think"]
        df_t       = eval_dfs.get(think_name)

        if solo:
            header = (
                f'<tr><th>Metric</th>'
                f'<th style="color:{ct}">🧠 {think_name}'
                f' <span style="font-size:.75rem;opacity:.6">(thinking only — no pair)</span></th></tr>'
            )
            rows_html = ""
            last_section = None
            for col_key, label, section in METRICS_META:
                if section != last_section:
                    rows_html += f'<tr class="section-header"><td colspan="2">{section}</td></tr>'
                    last_section = section
                avg_t        = avg_metric(df_t, col_key) if df_t is not None else None
                txt_t, cls_t = fmt_avg(col_key, avg_t)
                rows_html   += f"<tr><td class='metric-name'>{label}</td><td class='{cls_t}'>{txt_t}</td></tr>"
            n_t = len(df_t) if df_t is not None else 0
            html += f"""
        <h2 class="pair-title">{pair["label"]}</h2>
        <p class="pair-subtitle">{n_t} MCQs — thinking only (always thinks via internal &lt;think&gt; tags)</p>
        <table class="summary-table"><thead>{header}</thead><tbody>{rows_html}</tbody></table>"""
        else:
            nothink_name = pair["nothink"]["name"]
            cn           = pair["color_nothink"]
            df_nt        = eval_dfs.get(nothink_name)

            header = (
                f'<tr><th>Metric</th>'
                f'<th style="color:{ct}">🧠 {think_name}</th>'
                f'<th style="color:{cn}">⚡ {nothink_name}</th>'
                f'<th>Delta</th></tr>'
            )
            rows_html = ""
            last_section = None
            for col_key, label, section in METRICS_META:
                if section != last_section:
                    rows_html += f'<tr class="section-header"><td colspan="4">{section}</td></tr>'
                    last_section = section

                avg_t  = avg_metric(df_t,  col_key) if df_t  is not None else None
                avg_nt = avg_metric(df_nt, col_key) if df_nt is not None else None
                txt_t,  cls_t  = fmt_avg(col_key, avg_t)
                txt_nt, cls_nt = fmt_avg(col_key, avg_nt)

                delta_html = "<td class='na'>—</td>"
                if avg_t is not None and avg_nt is not None and col_key != "ambiguity":
                    delta = avg_t - avg_nt
                    delta_pct = delta * 100 if col_key not in ("originality", "relevance", "readability", "distractors_quality") else None
                    dtxt = f"{delta_pct:+.1f}pp" if delta_pct is not None else f"{delta:+.2f}"
                    dcls = "good" if delta > 0 else ("bad" if delta < 0 else "neutral")
                    delta_html = f"<td class='{dcls}'>{dtxt}</td>"

                rows_html += (
                    f"<tr><td class='metric-name'>{label}</td>"
                    f"<td class='{cls_t}'>{txt_t}</td>"
                    f"<td class='{cls_nt}'>{txt_nt}</td>"
                    f"{delta_html}</tr>"
                )

            n_t  = len(df_t)  if df_t  is not None else 0
            n_nt = len(df_nt) if df_nt is not None else 0
            html += f"""
        <h2 class="pair-title">{pair["label"]}</h2>
        <p class="pair-subtitle">{n_t} MCQs (think) · {n_nt} MCQs (no think)</p>
        <table class="summary-table"><thead>{header}</thead><tbody>{rows_html}</tbody></table>"""
    return html


def build_sample_cards(pairs: list, eval_dfs: dict, sample_ids: list[str]) -> str:
    """Per-LISA detail for a sample of rows. Solo pairs render as single column."""
    html = ""
    for pair in pairs:
        think_name = pair["think"]["name"]
        solo       = pair["nothink"] is None
        ct         = pair["color_think"]

        df_t  = eval_dfs.get(think_name)
        df_nt = eval_dfs.get(pair["nothink"]["name"]) if not solo else None

        if df_t is None and df_nt is None:
            continue

        indexed_t  = {str(r['lisa_id']): r for _, r in df_t.iterrows()}  if df_t  is not None else {}
        indexed_nt = {str(r['lisa_id']): r for _, r in df_nt.iterrows()} if df_nt is not None else {}

        html += f'<h2 class="pair-title">{pair["label"]} — sample cards</h2>'

        for lisa_id in sample_ids:
            row_t  = indexed_t.get(lisa_id)
            row_nt = indexed_nt.get(lisa_id) if not solo else None
            if row_t is None and row_nt is None:
                continue

            ref_row    = row_t if row_t is not None else row_nt
            folder     = ref_row.get('folder', '')
            content    = str(ref_row.get('content_raw', ''))[:300].replace('<', '&lt;').replace('>', '&gt;')
            if len(str(ref_row.get('content_raw', ''))) > 300:
                content += "…"

            cols_html  = _side_col(row_t, think_name, ct, "🧠 think")
            if not solo:
                cn         = pair["color_nothink"]
                cols_html += _side_col(row_nt, pair["nothink"]["name"], cn, "⚡ no think")

            grid_style = "grid-template-columns: 1fr;" if solo else ""
            html += f"""
            <div class="card">
              <div class="card-header">
                <span class="folder-badge">{folder}</span>
                <span class="lisa-id">{lisa_id}</span>
              </div>
              <details class="lisa-details">
                <summary>LISA source content</summary>
                <pre class="lisa-content">{content}</pre>
              </details>
              <div class="model-columns" style="{grid_style}">
                {cols_html}
              </div>
            </div>"""

    return html


def _side_col(row, model_name: str, color: str, badge: str) -> str:
    if row is None:
        return f'<div class="model-col" style="border-top:3px solid {color}"><p class="error">Generation failed</p></div>'

    correct = str(row.get('correct_option', '?')).lower()
    options_html = ""
    for opt in ['a', 'b', 'c', 'd']:
        text    = row.get(f'option_{opt}', '')
        comment = row.get(f'option_{opt}_comment', '')
        is_cor  = opt == correct
        cls     = "option correct-option" if is_cor else "option"
        badge_c = '<span class="correct-badge">✓</span>' if is_cor else ''
        options_html += f"""
        <div class="{cls}">
          <span class="opt-letter">{opt.upper()}.</span> {text} {badge_c}
          {f'<div class="opt-comment">{comment}</div>' if comment else ''}
        </div>"""

    metrics_html = ""
    for col_key, label, _ in METRICS_META:
        if col_key == "answerability_correct":
            gpt = str(row.get('gpt_answer', '')).strip().lower()
            val = gpt == correct
        elif col_key not in row:
            continue
        else:
            val = row[col_key]

        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue

        # Format
        if col_key in ("originality", "relevance", "ambiguity"):
            txt = f"{float(val):.2f}"
            if col_key == "ambiguity":
                cls = "neutral"
            else:
                cls = "good" if float(val) >= 0.5 else "bad"
        elif col_key == "readability":
            txt = f"{float(val):.1f}"
            cls = "good" if float(val) >= 12 else "bad"
        elif col_key == "is_question":
            b = bool(val)
            txt, cls = ("Yes ✓", "good") if b else ("No ✗", "bad")
        elif col_key == "starts_with_negation":
            b = bool(val)
            txt, cls = ("No ✓", "good") if not b else ("Yes ✗", "bad")
        elif col_key == "answerability_correct":
            txt, cls = ("Correct ✓", "good") if val else ("Wrong ✗", "bad")
        elif col_key == "disclosure":
            s = str(val).lower()
            txt, cls = ("Disclosed ✗", "bad") if s == "true" else ("Safe ✓", "good")
        elif col_key == "distractors_quality":
            try:
                detail_raw = row.get("distractors_quality_detail")
                detail = json.loads(detail_raw) if isinstance(detail_raw, str) else detail_raw
                if isinstance(detail, dict):
                    avg = detail.get('avg', 0)
                    scores = detail.get('scores', [])
                else:
                    scores = [d["score"] for d in detail]
                    avg = sum(scores) / len(scores)
                cls = "good" if avg >= 4 else ("warn" if avg >= 3 else "bad")
                txt = f"avg {avg:.1f}/5  [{' / '.join(str(s) for s in scores)}]"
            except Exception:
                txt, cls = "N/A", "na"
        else:
            txt, cls = str(val), "neutral"

        metrics_html += f"""
        <div class="metric-row">
          <span class="metric-label">{label}</span>
          <span class="metric-value {cls}">{txt}</span>
        </div>"""

    q_comment = row.get('question_comment', '')
    return f"""
    <div class="model-col" style="border-top:3px solid {color}">
      <div class="model-col-header" style="color:{color}">{badge} {model_name}</div>
      <p class="question-text">{row.get('question', '')}</p>
      {f'<div class="question-comment">{q_comment}</div>' if q_comment else ''}
      <div class="options-list">{options_html}</div>
      <div class="metrics-list">{metrics_html}</div>
    </div>"""


# ---------------------------------------------------------------------------
# Pick sample LISA IDs for cards
# ---------------------------------------------------------------------------

all_ids = df_lisa['id'].astype(str).tolist()
random.seed(RANDOM_SEED)
sample_ids = random.sample(all_ids, min(SAMPLE_CARDS, len(all_ids)))

# ---------------------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------------------

summary_html = build_global_summary(PAIRS, eval_dfs)
cards_html   = build_sample_cards(PAIRS, eval_dfs, sample_ids)
timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M")

n_total = sum(len(r) for r in raw_results.values())

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MCQ Benchmark — Thinking vs Non-Thinking Models</title>
<style>
  :root {{
    --radius: 10px;
    --shadow: 0 2px 8px rgba(0,0,0,.08);
    --good:    #16a34a;
    --bad:     #dc2626;
    --warn:    #d97706;
    --neutral: #374151;
    --na:      #9ca3af;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #f8fafc; color: #1e293b; padding: 2rem; max-width: 1400px; margin: auto; }}

  h1  {{ font-size: 1.7rem; font-weight: 700; margin-bottom: .25rem; }}
  .subtitle {{ color: #64748b; font-size: .95rem; margin-bottom: 2rem; }}
  .pair-title {{ font-size: 1.2rem; font-weight: 700; margin: 2.5rem 0 .25rem; color: #334155;
                 border-left: 4px solid #0ea5e9; padding-left: .75rem; }}
  .pair-subtitle {{ font-size: .85rem; color: #94a3b8; margin-bottom: .75rem; }}

  /* Summary table */
  .summary-table {{ width: 100%; border-collapse: collapse; background: white;
                    border-radius: var(--radius); overflow: hidden;
                    box-shadow: var(--shadow); margin-bottom: 1.5rem; }}
  .summary-table th {{ background: #1e293b; color: white; padding: .75rem 1rem;
                       text-align: left; font-size: .9rem; }}
  .summary-table td {{ padding: .6rem 1rem; border-bottom: 1px solid #f1f5f9; }}
  .summary-table tr:last-child td {{ border-bottom: none; }}
  .summary-table tr:hover td {{ background: #f8fafc; }}
  .summary-table .section-header td {{ background: #f1f5f9; color: #64748b;
    font-size: .78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .05em; padding: .35rem 1rem; }}
  .metric-name {{ font-weight: 500; color: #374151; }}

  .good    {{ color: var(--good);    font-weight: 600; }}
  .bad     {{ color: var(--bad);     font-weight: 600; }}
  .warn    {{ color: var(--warn);    font-weight: 600; }}
  .neutral {{ color: var(--neutral); }}
  .na      {{ color: var(--na); }}

  /* Cards */
  .card {{ background: white; border-radius: var(--radius); box-shadow: var(--shadow);
           margin-bottom: 1.5rem; overflow: hidden; }}
  .card-header {{ display: flex; align-items: center; gap: .75rem;
                  padding: .85rem 1.25rem; background: #1e293b; color: white; }}
  .folder-badge {{ background: #0ea5e9; padding: .2rem .6rem; border-radius: 4px;
                   font-size: .8rem; font-weight: 700; }}
  .lisa-id {{ font-size: .9rem; color: #94a3b8; }}

  .lisa-details {{ padding: .5rem 1.25rem; border-bottom: 1px solid #f1f5f9; }}
  .lisa-details summary {{ cursor: pointer; font-size: .85rem; color: #64748b;
                            padding: .4rem 0; user-select: none; }}
  .lisa-content {{ font-size: .78rem; color: #475569; background: #f8fafc;
                   padding: .75rem; border-radius: 6px; margin-top: .5rem;
                   white-space: pre-wrap; word-break: break-word;
                   max-height: 200px; overflow-y: auto; }}

  .model-columns {{ display: grid; grid-template-columns: 1fr 1fr; }}
  .model-col {{ padding: 1.25rem; border-right: 1px solid #f1f5f9; }}
  .model-col:last-child {{ border-right: none; }}
  .model-col-header {{ font-weight: 700; font-size: .95rem; margin-bottom: .75rem; }}

  .question-text {{ font-size: .95rem; font-weight: 600; line-height: 1.5;
                    color: #1e293b; margin-bottom: .5rem; }}
  .question-comment {{ font-size: .82rem; color: #64748b; font-style: italic;
                        margin-bottom: .75rem; padding-left: .75rem;
                        border-left: 3px solid #e2e8f0; }}

  .options-list {{ margin-bottom: 1rem; }}
  .option {{ padding: .4rem .6rem; border-radius: 6px; font-size: .85rem;
             margin-bottom: .3rem; line-height: 1.4; }}
  .correct-option {{ background: #dcfce7; }}
  .opt-letter {{ font-weight: 700; color: #475569; }}
  .correct-badge {{ background: #16a34a; color: white; font-size: .7rem;
                    padding: .1rem .35rem; border-radius: 4px; margin-left: .3rem; }}
  .opt-comment {{ font-size: .76rem; color: #64748b; margin-top: .15rem;
                  padding-left: 1rem; font-style: italic; }}

  .metrics-list {{ border-top: 1px solid #f1f5f9; padding-top: .6rem; }}
  .metric-row {{ display: flex; gap: .5rem; padding: .25rem 0;
                 border-bottom: 1px dotted #f1f5f9; align-items: baseline; }}
  .metric-row:last-child {{ border-bottom: none; }}
  .metric-label {{ font-size: .78rem; color: #94a3b8; flex: 0 0 130px; }}
  .metric-value {{ font-size: .83rem; font-weight: 600; }}

  .error {{ color: #dc2626; font-style: italic; font-size: .9rem; padding: 1rem 0; }}

  @media (max-width: 800px) {{
    .model-columns {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<h1>MCQ Benchmark — Thinking vs Non-Thinking Models</h1>
<p class="subtitle">
  Generated {timestamp} &nbsp;·&nbsp;
  {N_LISA_ROWS} LISA rows &nbsp;·&nbsp;
  6 pairs (12 model sides) &nbsp;·&nbsp;
  {n_total} total MCQs &nbsp;·&nbsp;
  {SAMPLE_CARDS} sampled cards shown
</p>

<h2 style="font-size:1.4rem;margin-bottom:.5rem;margin-top:2rem">Summary — per pair</h2>
{summary_html}

<h2 style="font-size:1.4rem;margin-bottom:.5rem;margin-top:3rem">Sample cards ({SAMPLE_CARDS} random LISA rows)</h2>
{cards_html}

</body>
</html>
"""

os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"\nReport written to: {OUTPUT_HTML}")
