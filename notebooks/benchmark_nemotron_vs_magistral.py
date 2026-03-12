"""
benchmark_nemotron_vs_magistral.py
-----------------------------------
Generates 10 MCQs with Nemotron and Magistral from LISA sheets,
evaluates them with all metrics, and outputs a comparison HTML report.

Usage:
    cd notebooks
    python benchmark_nemotron_vs_magistral.py
"""

import sys
import os
import json
import math
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'page', 'backend'))

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'src', 'page', 'backend', '.env'))

from api.utils.generate_mcq_GPU import generate_mcq, _HF_PROMPT_TEMPLATE, validate_mcq, shuffle_distractors
from eval.eval_dataframe import eval_dataframe

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

MODELS = {
    "Nemotron 30B": "hf.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q8_0",
    "Magistral 24B": "magistral:latest",
}

N_LISA_ROWS = 10

LISA_CSV    = os.path.join(os.path.dirname(__file__), '..', 'src', 'page', 'backend', 'data', 'lisa_sheets.csv')
PROMPTS_JSON = os.path.join(os.path.dirname(__file__), '..', 'src', 'page', 'backend', 'eval', 'prompts.json')
OUTPUT_HTML = os.path.join(os.path.dirname(__file__), '..', 'docs', 'benchmark_nemotron_vs_magistral.html')

# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------

with open(PROMPTS_JSON) as f:
    prompts = json.load(f)

df_lisa = pd.read_csv(LISA_CSV, nrows=N_LISA_ROWS)
print(f"Loaded {len(df_lisa)} LISA rows")

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

raw_results: dict[str, list[dict]] = {}

for model_label, model_name in MODELS.items():
    print(f"\n[GEN] {model_label} ({model_name})")
    rows = []
    for _, lisa_row in df_lisa.iterrows():
        prompt = _HF_PROMPT_TEMPLATE.format(content=lisa_row['content_raw'])
        try:
            mcq_json = generate_mcq(prompt, model_name)
            mcq = validate_mcq(mcq_json)
            if mcq:
                mcq = shuffle_distractors(mcq)
                rows.append({
                    'lisa_id':      lisa_row['id'],
                    'folder':       lisa_row['folder'],
                    'content_raw':  lisa_row['content_raw'],
                    **mcq.model_dump(),
                })
                print(f"  ✓ {lisa_row['id']}")
            else:
                print(f"  ✗ {lisa_row['id']} — validation failed")
        except Exception as e:
            print(f"  ✗ {lisa_row['id']} — {e}")
    raw_results[model_label] = rows

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

eval_dfs: dict[str, pd.DataFrame] = {}

for model_label, rows in raw_results.items():
    if not rows:
        print(f"[EVAL] {model_label}: no rows to evaluate, skipping")
        continue
    print(f"\n[EVAL] {model_label} — {len(rows)} MCQs")
    df = pd.DataFrame(rows)
    df = eval_dataframe(
        df,
        openai_key=OPENAI_KEY,
        answerability_system_prompt=prompts['answerability_prompt'],
        disclosure_system_prompt=prompts['disclosure_prompt'],
        difficulty_system_prompt=prompts['difficulty_prompt'],
        distractors_quality_system_prompt=prompts['distractors_quality_prompt'],
        output_file_path=f'/tmp/eval_{model_label.replace(" ", "_")}.csv',
    )
    eval_dfs[model_label] = df
    print(f"  done — columns: {list(df.columns)}")

# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    list(MODELS.keys())[0]: ("#0ea5e9", "#e0f2fe"),   # blue
    list(MODELS.keys())[1]: ("#8b5cf6", "#ede9fe"),   # purple
}


def fmt_score(val, metric: str) -> tuple[str, str]:
    """Return (display_text, css_class) for a metric value."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—", "na"

    if metric == "originality":
        v = float(val)
        cls = "good" if v >= 0.75 else "bad"
        return f"{v:.2f}", cls

    if metric == "readability":
        v = float(val)
        cls = "good" if v >= 12 else "bad"
        return f"{v:.1f}", cls

    if metric == "starts_with_negation":
        b = bool(val)
        return ("Yes ✗", "bad") if b else ("No ✓", "good")

    if metric == "is_question":
        b = bool(val)
        return ("Yes ✓", "good") if b else ("No ✗", "bad")

    if metric == "relevance":
        v = float(val)
        cls = "good" if v >= 0.5 else "bad"
        return f"{v:.2f}", cls

    if metric == "ambiguity":
        v = float(val)
        cls = "good" if v < 0.5 else "bad"
        return f"{v:.2f}", cls

    if metric == "gpt_answer":
        return str(val).upper(), "neutral"

    if metric == "answerability_correct":
        b = bool(val)
        return ("Correct ✓", "good") if b else ("Wrong ✗", "bad")

    if metric == "disclosure":
        s = str(val).lower()
        is_disclosed = s == "true"
        return ("Disclosed ✗", "bad") if is_disclosed else ("Safe ✓", "good")

    if metric == "difficulty":
        s = str(val)
        cls = {"low": "warn", "med": "neutral", "high": "good"}.get(s.lower(), "neutral")
        return s.capitalize(), cls

    if metric == "distractors_quality":
        try:
            items = json.loads(val) if isinstance(val, str) else val
            scores = [int(d["score"]) for d in items]
            avg = sum(scores) / len(scores)
            cls = "good" if avg >= 4 else ("warn" if avg >= 3 else "bad")
            return f"{avg:.1f} ({'/'.join(str(s) for s in scores)})", cls
        except Exception:
            return str(val)[:20], "neutral"

    return str(val), "neutral"


def distractor_detail_html(val) -> str:
    """Render distractor quality JSON as small inline detail."""
    try:
        items = json.loads(val) if isinstance(val, str) else val
        letters = ["b", "c", "d"]  # distractors (a is sometimes correct)
        parts = []
        for i, d in enumerate(items):
            score = d.get("score", "?")
            justif = d.get("justif", "")
            color = "#16a34a" if score >= 4 else ("#d97706" if score >= 3 else "#dc2626")
            parts.append(
                f'<span style="color:{color};font-weight:600">{score}/5</span>'
                f' <span style="color:#6b7280;font-size:0.8em">{justif}</span>'
            )
        return "<br>".join(parts)
    except Exception:
        return str(val)


METRICS_META = [
    ("is_question",           "Is a Question",       "Section A"),
    ("starts_with_negation",  "Starts with Negation","Section A"),
    ("originality",           "Originality",         "Section A"),
    ("readability",           "Readability (FK)",    "Section A"),
    ("relevance",             "Relevance",           "Section B"),
    ("ambiguity",             "Ambiguity",           "Section B"),
    ("answerability_correct", "Answerability",       "Section B"),
    ("disclosure",            "Disclosure",          "Section B"),
    ("difficulty",            "Difficulty",          "Section B"),
    ("distractors_quality",   "Distractor Quality",  "Section B"),
]


def build_summary(dfs: dict[str, pd.DataFrame]) -> str:
    """Summary table: one row per metric, one column per model."""
    model_names = list(dfs.keys())

    def avg_metric(df, metric):
        if metric not in df.columns:
            return None
        col = df[metric]
        if metric in ("originality", "readability", "relevance", "ambiguity"):
            return col.dropna().astype(float).mean()
        if metric in ("is_question",):
            return col.dropna().astype(bool).mean()
        if metric == "starts_with_negation":
            return col.dropna().astype(bool).mean()
        if metric == "answerability_correct":
            return col.dropna().astype(bool).mean()
        if metric == "disclosure":
            return (col.dropna().astype(str).str.lower() == "true").mean()
        if metric == "distractors_quality":
            scores = []
            for v in col.dropna():
                try:
                    items = json.loads(v) if isinstance(v, str) else v
                    scores.append(sum(d["score"] for d in items) / len(items))
                except Exception:
                    pass
            return sum(scores) / len(scores) if scores else None
        return None

    rows_html = ""
    last_section = None
    for col_key, label, section in METRICS_META:
        if section != last_section:
            rows_html += f'<tr class="section-header"><td colspan="{1+len(model_names)}">{section}</td></tr>'
            last_section = section

        cells = f"<td class='metric-name'>{label}</td>"
        for model_label in model_names:
            df = dfs.get(model_label)
            if df is None or col_key not in df.columns:
                cells += "<td class='na'>—</td>"
                continue
            avg = avg_metric(df, col_key)
            if avg is None:
                cells += "<td class='na'>—</td>"
                continue

            # Display
            if col_key in ("originality", "relevance", "ambiguity"):
                txt = f"{avg:.2f}"
                cls = "good" if (
                    (col_key != "ambiguity" and avg >= 0.5) or
                    (col_key == "ambiguity" and avg < 0.5)
                ) else "bad"
            elif col_key == "readability":
                txt = f"{avg:.1f}"
                cls = "good" if avg >= 12 else "bad"
            elif col_key in ("is_question", "answerability_correct"):
                txt = f"{avg*100:.0f}%"
                cls = "good" if avg >= 0.8 else ("warn" if avg >= 0.5 else "bad")
            elif col_key == "starts_with_negation":
                txt = f"{avg*100:.0f}%"
                cls = "bad" if avg > 0.1 else "good"
            elif col_key == "disclosure":
                txt = f"{avg*100:.0f}%"
                cls = "bad" if avg > 0.2 else "good"
            elif col_key == "distractors_quality":
                txt = f"{avg:.2f}/5"
                cls = "good" if avg >= 4 else ("warn" if avg >= 3 else "bad")
            else:
                txt = str(avg)
                cls = "neutral"

            cells += f"<td class='{cls}'>{txt}</td>"
        rows_html += f"<tr>{cells}</tr>"

    header_cells = "<th>Metric</th>" + "".join(
        f'<th style="color:{MODEL_COLORS[m][0]}">{m}</th>' for m in model_names
    )
    return f"""
    <table class="summary-table">
      <thead><tr>{header_cells}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """


def build_detail_cards(dfs: dict[str, pd.DataFrame]) -> str:
    """Per-LISA-row detail section showing both models' MCQ + scores."""
    model_names = list(dfs.keys())
    html = ""

    # Index by lisa_id
    indexed = {}
    for model_label, df in dfs.items():
        if 'lisa_id' in df.columns:
            for _, row in df.iterrows():
                lid = row['lisa_id']
                indexed.setdefault(lid, {})[model_label] = row

    for lisa_id, model_rows in indexed.items():
        folder = next(iter(model_rows.values())).get('folder', '')
        content_raw = next(iter(model_rows.values())).get('content_raw', '')
        # Truncate LISA content for display
        content_preview = content_raw[:300].replace('<', '&lt;').replace('>', '&gt;')
        if len(content_raw) > 300:
            content_preview += "…"

        html += f"""
        <div class="card">
          <div class="card-header">
            <span class="folder-badge">{folder}</span>
            <span class="lisa-id">{lisa_id}</span>
          </div>
          <details class="lisa-details">
            <summary>LISA source content</summary>
            <pre class="lisa-content">{content_preview}</pre>
          </details>
          <div class="model-columns">
        """

        for model_label in model_names:
            row = model_rows.get(model_label)
            color, bg = MODEL_COLORS[model_label]
            if row is None:
                html += f'<div class="model-col" style="border-top:3px solid {color}"><p class="error">Generation failed</p></div>'
                continue

            correct = str(row.get('correct_option', '?')).lower()
            options_html = ""
            for opt in ['a', 'b', 'c', 'd']:
                text = row.get(f'option_{opt}', '')
                comment = row.get(f'option_{opt}_comment', '')
                is_correct = opt == correct
                cls = "option correct-option" if is_correct else "option"
                badge = f'<span class="correct-badge">✓ correct</span>' if is_correct else ''
                options_html += f"""
                <div class="{cls}">
                  <span class="opt-letter">{opt.upper()}.</span> {text} {badge}
                  {f'<div class="opt-comment">{comment}</div>' if comment else ''}
                </div>"""

            metrics_html = ""
            for col_key, label, _ in METRICS_META:
                if col_key == "answerability_correct":
                    gpt = str(row.get('gpt_answer', '')).lower()
                    val = gpt == correct
                    col_key_display = "answerability_correct"
                    display, cls = fmt_score(val, "answerability_correct")
                elif col_key not in row:
                    continue
                else:
                    val = row[col_key]
                    display, cls = fmt_score(val, col_key)

                extra = ""
                if col_key == "distractors_quality" and val and str(val) != "nan":
                    extra = f'<div class="distractor-detail">{distractor_detail_html(val)}</div>'

                metrics_html += f"""
                <div class="metric-row">
                  <span class="metric-label">{label}</span>
                  <span class="metric-value {cls}">{display}</span>
                  {extra}
                </div>"""

            q_comment = row.get('question_comment', '')
            html += f"""
            <div class="model-col" style="border-top:3px solid {color};background:{bg}10">
              <div class="model-col-header" style="color:{color}">{model_label}</div>
              <p class="question-text">{row.get('question','')}</p>
              {f'<div class="question-comment">{q_comment}</div>' if q_comment else ''}
              <div class="options-list">{options_html}</div>
              <div class="metrics-list">{metrics_html}</div>
            </div>"""

        html += "</div></div>"  # close model-columns + card

    return html


# ---------------------------------------------------------------------------
# Add answerability_correct column
# ---------------------------------------------------------------------------

for model_label, df in eval_dfs.items():
    if 'gpt_answer' in df.columns and 'correct_option' in df.columns:
        df['answerability_correct'] = (
            df['gpt_answer'].astype(str).str.strip().str.lower() ==
            df['correct_option'].astype(str).str.strip().str.lower()
        )

# ---------------------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------------------

summary_html  = build_summary(eval_dfs)
detail_html   = build_detail_cards(eval_dfs)
timestamp     = datetime.now().strftime("%Y-%m-%d %H:%M")
model_labels  = list(MODELS.keys())

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MCQ Benchmark — {model_labels[0]} vs {model_labels[1]}</title>
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
  body {{ font-family: system-ui, sans-serif; background: #f8fafc; color: #1e293b; padding: 2rem; }}

  h1 {{ font-size: 1.7rem; font-weight: 700; margin-bottom: .25rem; }}
  .subtitle {{ color: #64748b; font-size: .95rem; margin-bottom: 2rem; }}
  h2 {{ font-size: 1.2rem; font-weight: 600; margin: 2rem 0 1rem; color: #334155; }}

  /* Summary table */
  .summary-table {{ width: 100%; border-collapse: collapse; background: white;
                    border-radius: var(--radius); overflow: hidden;
                    box-shadow: var(--shadow); margin-bottom: 2.5rem; }}
  .summary-table th {{ background: #1e293b; color: white; padding: .75rem 1rem;
                       text-align: left; font-size: .95rem; }}
  .summary-table td {{ padding: .65rem 1rem; border-bottom: 1px solid #f1f5f9; }}
  .summary-table tr:last-child td {{ border-bottom: none; }}
  .summary-table tr:hover td {{ background: #f8fafc; }}
  .summary-table .section-header td {{ background: #f1f5f9; color: #64748b;
                                        font-size: .8rem; font-weight: 700;
                                        text-transform: uppercase; letter-spacing: .05em;
                                        padding: .4rem 1rem; }}
  .metric-name {{ font-weight: 500; color: #374151; }}

  /* Score classes */
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
                   white-space: pre-wrap; word-break: break-word; max-height: 200px;
                   overflow-y: auto; }}

  .model-columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; }}
  .model-col {{ padding: 1.25rem; border-right: 1px solid #f1f5f9; }}
  .model-col:last-child {{ border-right: none; }}
  .model-col-header {{ font-weight: 700; font-size: 1rem; margin-bottom: .75rem; }}

  .question-text {{ font-size: .97rem; font-weight: 600; line-height: 1.5;
                    color: #1e293b; margin-bottom: .5rem; }}
  .question-comment {{ font-size: .82rem; color: #64748b; font-style: italic;
                        margin-bottom: .75rem; padding-left: .75rem;
                        border-left: 3px solid #e2e8f0; }}

  .options-list {{ margin-bottom: 1rem; }}
  .option {{ padding: .4rem .6rem; border-radius: 6px; font-size: .88rem;
             margin-bottom: .35rem; line-height: 1.4; }}
  .correct-option {{ background: #dcfce7; }}
  .opt-letter {{ font-weight: 700; color: #475569; }}
  .correct-badge {{ background: #16a34a; color: white; font-size: .7rem;
                    padding: .1rem .4rem; border-radius: 4px; margin-left: .4rem; }}
  .opt-comment {{ font-size: .78rem; color: #64748b; margin-top: .2rem;
                  padding-left: 1.1rem; font-style: italic; }}

  .metrics-list {{ border-top: 1px solid #f1f5f9; padding-top: .75rem; }}
  .metric-row {{ display: flex; flex-wrap: wrap; align-items: baseline;
                 gap: .5rem; padding: .3rem 0;
                 border-bottom: 1px dotted #f1f5f9; }}
  .metric-row:last-child {{ border-bottom: none; }}
  .metric-label {{ font-size: .8rem; color: #94a3b8; flex: 0 0 140px; }}
  .metric-value {{ font-size: .85rem; font-weight: 600; }}
  .distractor-detail {{ flex: 1 0 100%; font-size: .78rem; padding-left: 140px;
                         padding-top: .15rem; }}

  .error {{ color: #dc2626; font-style: italic; font-size: .9rem; padding: 1rem 0; }}

  @media (max-width: 700px) {{
    .model-columns {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<h1>MCQ Benchmark — {model_labels[0]} vs {model_labels[1]}</h1>
<p class="subtitle">Generated {timestamp} · {N_LISA_ROWS} LISA rows · all metrics enabled</p>

<h2>Summary</h2>
{summary_html}

<h2>Per-Question Detail</h2>
{detail_html}

</body>
</html>
"""

os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"\nReport written to: {OUTPUT_HTML}")
