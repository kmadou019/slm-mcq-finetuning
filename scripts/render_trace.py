#!/usr/bin/env python3
"""
Convert trace.jsonl → trace.html (self-contained, no external deps).

Usage:
  python scripts/render_trace.py                          # reads data/optimized_prompts/trace.jsonl
  python scripts/render_trace.py --trace path/trace.jsonl
"""

import argparse
import difflib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR   = SCRIPT_DIR.parent


# ─────────────────────────────────────────────────────────────────
# Parse JSONL → nested structure
# ─────────────────────────────────────────────────────────────────

def load_events(path: Path) -> list[dict]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def build_structure(events: list[dict]) -> dict:
    """Group events into {models: [{save_name, sheets: [{attempts: [...]}]}]}"""
    models: dict[str, dict] = {}
    global_summary = None

    for ev in events:
        t = ev.get("event", "")
        sn = ev.get("save_name", "")

        if t == "model_start":
            models[sn] = {
                "save_name": sn,
                "model_id": ev.get("model_id", ""),
                "n_sheets": ev.get("n_sheets", 0),
                "max_attempts": ev.get("max_attempts", 0),
                "load_duration_s": None,
                "total_duration_s": None,
                "passed": None,
                "total": None,
                "sheets": {},
            }

        elif t == "model_loaded" and sn in models:
            models[sn]["load_duration_s"] = ev.get("duration_s")

        elif t == "sheet_start" and sn in models:
            idx = ev.get("sheet_idx", 0)
            models[sn]["sheets"][idx] = {
                "sheet_idx": idx,
                "sheet_id": ev.get("sheet_id"),
                "passed": None,
                "n_attempts": None,
                "attempts": [],
            }

        elif t == "generation" and sn in models:
            idx = ev.get("sheet_idx", 0)
            attempt = ev.get("attempt", 0)
            sheet = models[sn]["sheets"].setdefault(idx, {"sheet_idx": idx, "attempts": []})
            while len(sheet["attempts"]) <= attempt:
                sheet["attempts"].append({})
            sheet["attempts"][attempt].update({
                "gen_duration_s": ev.get("duration_s"),
                "gen_success": ev.get("success"),
                "question": ev.get("question"),
                "correct_option": ev.get("correct_option"),
            })

        elif t == "b3_eval" and sn in models:
            idx = ev.get("sheet_idx", 0)
            attempt = ev.get("attempt", 0)
            sheet = models[sn]["sheets"].setdefault(idx, {"sheet_idx": idx, "attempts": []})
            while len(sheet["attempts"]) <= attempt:
                sheet["attempts"].append({})
            sheet["attempts"][attempt].update({
                "b3_duration_s": ev.get("duration_s"),
                "b3_passes": ev.get("passes"),
                "b3_scores": ev.get("scores", []),
                "b3_justifs": ev.get("justifs", []),
            })

        elif t == "claude_improve" and sn in models:
            idx = ev.get("sheet_idx", 0)
            attempt = ev.get("attempt", 0)
            sheet = models[sn]["sheets"].setdefault(idx, {"sheet_idx": idx, "attempts": []})
            while len(sheet["attempts"]) <= attempt:
                sheet["attempts"].append({})
            sheet["attempts"][attempt].update({
                "claude_duration_s": ev.get("duration_s"),
                "prompt_before_chars": ev.get("chars_before"),
                "prompt_after_chars": ev.get("chars_after"),
                "prompt_after": ev.get("prompt_after", ""),
            })

        elif t == "sheet_done" and sn in models:
            idx = ev.get("sheet_idx", 0)
            if idx in models[sn]["sheets"]:
                models[sn]["sheets"][idx]["passed"] = ev.get("passed")
                models[sn]["sheets"][idx]["n_attempts"] = ev.get("n_attempts")

        elif t == "model_done" and sn in models:
            models[sn]["total_duration_s"] = ev.get("duration_s")
            models[sn]["passed"] = ev.get("passed")
            models[sn]["total"] = ev.get("total")

        elif t == "global_summary":
            global_summary = ev

    return {"models": list(models.values()), "global": global_summary}


# ─────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────

def _score_badge(score: int) -> str:
    if score >= 5:
        cls, label = "score-5", "5"
    elif score >= 4:
        cls, label = "score-4", "4"
    elif score == 3:
        cls, label = "score-3", "3"
    else:
        cls, label = "score-low", str(score)
    return f'<span class="score-badge {cls}">{label}</span>'


def _pass_badge(passes: bool | None) -> str:
    if passes is None:
        return '<span class="badge badge-skip">—</span>'
    if passes:
        return '<span class="badge badge-pass">✅ pass</span>'
    return '<span class="badge badge-fail">❌ fail</span>'


def _prompt_diff_html(before: str, after: str) -> str:
    if not before or not after or before == after:
        return ""
    diff = list(difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile="avant", tofile="après", lineterm="", n=3,
    ))
    if not diff:
        return ""
    lines_html = []
    for line in diff:
        if line.startswith("+++") or line.startswith("---"):
            lines_html.append(f'<span class="diff-hdr">{_esc(line)}</span>')
        elif line.startswith("@@"):
            lines_html.append(f'<span class="diff-ctx">{_esc(line)}</span>')
        elif line.startswith("+"):
            lines_html.append(f'<span class="diff-add">{_esc(line)}</span>')
        elif line.startswith("-"):
            lines_html.append(f'<span class="diff-del">{_esc(line)}</span>')
        else:
            lines_html.append(f'<span class="diff-unchanged">{_esc(line)}</span>')
    inner = "\n".join(lines_html)
    return f'<pre class="diff-block">{inner}</pre>'


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ─────────────────────────────────────────────────────────────────
# HTML sections
# ─────────────────────────────────────────────────────────────────

def render_attempt(attempt: dict, attempt_idx: int, prev_prompt: str) -> str:
    gen_ok   = attempt.get("gen_success", False)
    question = attempt.get("question") or ""
    scores   = attempt.get("b3_scores", [])
    justifs  = attempt.get("b3_justifs", [])
    passes   = attempt.get("b3_passes")
    prompt_after = attempt.get("prompt_after", "")

    gen_dur   = attempt.get("gen_duration_s")
    b3_dur    = attempt.get("b3_duration_s")
    cl_dur    = attempt.get("claude_duration_s")

    parts = []

    # Generation row
    gen_status = f'<span class="ok">✓ {gen_dur:.1f}s</span>' if gen_ok else '<span class="fail">INVALIDE</span>'
    parts.append(f'<div class="row-label">Génération</div><div class="row-val">{gen_status}'
                 + (f' — <em>{_esc(question)}</em>' if question else "") + "</div>")

    # B3 row
    if scores:
        badges = " ".join(_score_badge(s) for s in scores)
        b3_pass_badge = _pass_badge(passes)
        jlines = "".join(f'<li>{_esc(j)}</li>' for j in justifs if j)
        justif_html = f'<ul class="justif-list">{jlines}</ul>' if jlines else ""
        dur_str = f' ({b3_dur:.1f}s)' if b3_dur else ""
        parts.append(
            f'<div class="row-label">B3 GPT-4o</div>'
            f'<div class="row-val">{b3_pass_badge} {badges}{dur_str}{justif_html}</div>'
        )

    # Claude improvement diff
    if prompt_after:
        diff_html = _prompt_diff_html(prev_prompt, prompt_after)
        dur_str = f" ({cl_dur:.1f}s)" if cl_dur else ""
        char_diff = attempt.get("prompt_after_chars", 0) - attempt.get("prompt_before_chars", 0)
        sign = "+" if char_diff >= 0 else ""
        collapse_id = f"diff-{id(attempt)}"
        parts.append(
            f'<div class="row-label">Claude{dur_str}</div>'
            f'<div class="row-val">'
            f'<details id="{collapse_id}"><summary>Diff prompt ({sign}{char_diff} chars)</summary>'
            + diff_html
            + '</details>'
            + f'<details><summary>Prompt complet après</summary>'
            f'<pre class="prompt-pre">{_esc(prompt_after)}</pre></details>'
            + "</div>"
        )

    rows = "\n".join(
        f'<div class="attempt-row">{p}</div>' for p in parts
    )
    return f'<div class="attempt-block"><h5>Tentative {attempt_idx + 1}</h5><div class="attempt-grid">{rows}</div></div>'


def render_sheet(sheet: dict, all_prompts: list[str]) -> str:
    idx       = sheet.get("sheet_idx", 0)
    sheet_id  = sheet.get("sheet_id", idx)
    passed    = sheet.get("passed")
    n_att     = sheet.get("n_attempts", len(sheet.get("attempts", [])))
    badge     = _pass_badge(passed)
    attempts  = sheet.get("attempts", [])

    attempts_html = ""
    prev_prompt = all_prompts[0] if all_prompts else ""
    for i, att in enumerate(attempts):
        attempts_html += render_attempt(att, i, prev_prompt)
        if att.get("prompt_after"):
            prev_prompt = att["prompt_after"]

    return (
        f'<details class="sheet-details">'
        f'<summary>Fiche {idx + 1} &nbsp; <code>id={_esc(str(sheet_id))}</code>'
        f'&nbsp; {badge} &nbsp; <small>{n_att} tentative(s)</small></summary>'
        f'<div class="sheet-body">{attempts_html}</div>'
        f'</details>'
    )


def render_model(model: dict, initial_prompt: str) -> str:
    sn       = model["save_name"]
    mid      = model["model_id"]
    passed   = model.get("passed")
    total    = model.get("total")
    dur      = model.get("total_duration_s")
    load_dur = model.get("load_duration_s")
    sheets   = sorted(model["sheets"].values(), key=lambda s: s.get("sheet_idx", 0))

    rate_str = ""
    if passed is not None and total:
        pct = round(100 * passed / total)
        color = "#27ae60" if pct >= 70 else "#e67e22" if pct >= 40 else "#e74c3c"
        rate_str = (
            f'<div class="model-rate">'
            f'<div class="rate-bar-bg"><div class="rate-bar" style="width:{pct}%;background:{color}"></div></div>'
            f'<span style="color:{color};font-weight:bold">{passed}/{total} ({pct}%)</span>'
            f'</div>'
        )

    meta = []
    if load_dur: meta.append(f"Chargement : {load_dur:.0f}s")
    if dur:      meta.append(f"Durée totale : {dur:.0f}s")
    meta_html = " &nbsp;·&nbsp; ".join(meta)

    all_prompts = [initial_prompt]
    sheets_html = "".join(render_sheet(s, all_prompts) for s in sheets)

    initial_html = (
        f'<details class="prompt-init">'
        f'<summary>Prompt initial</summary>'
        f'<pre class="prompt-pre">{_esc(initial_prompt)}</pre>'
        f'</details>'
    )

    return (
        f'<section class="model-section">'
        f'<h2>{_esc(sn)} <small class="model-id">{_esc(mid)}</small></h2>'
        f'{rate_str}'
        f'<p class="model-meta">{meta_html}</p>'
        f'{initial_html}'
        f'<div class="sheets-container">{sheets_html}</div>'
        f'</section>'
    )


def render_global(data: dict, trace_path: Path) -> str:
    g = data.get("global") or {}
    ts    = g.get("ts", "")[:19].replace("T", " ")
    total = g.get("total_sheets", 0)
    passed = g.get("total_passed", 0)
    elapsed = g.get("total_elapsed_s", 0)
    models = data["models"]

    rows = ""
    for m in models:
        p, t = m.get("passed") or 0, m.get("total") or 0
        pct = round(100 * p / t) if t else 0
        dur = m.get("total_duration_s")
        dur_str = f"{dur:.0f}s" if dur else "—"
        color = "#27ae60" if pct >= 70 else "#e67e22" if pct >= 40 else "#e74c3c"
        rows += (
            f'<tr>'
            f'<td><strong>{_esc(m["save_name"])}</strong></td>'
            f'<td><small>{_esc(m["model_id"])}</small></td>'
            f'<td style="color:{color};font-weight:bold">{p}/{t} ({pct}%)</td>'
            f'<td>{dur_str}</td>'
            f'</tr>'
        )
    total_pct = round(100 * passed / total) if total else 0

    return (
        f'<div class="global-summary">'
        f'<h1>Rapport d\'optimisation de prompt</h1>'
        f'<p class="trace-meta">Trace : <code>{_esc(str(trace_path))}</code>'
        + (f' &nbsp;·&nbsp; {ts}' if ts else "") +
        f' &nbsp;·&nbsp; Durée totale : {elapsed:.0f}s</p>'
        f'<table class="summary-table"><thead>'
        f'<tr><th>Modèle</th><th>HF ID</th><th>Fiches passées</th><th>Durée</th></tr>'
        f'</thead><tbody>{rows}'
        f'<tr class="total-row"><td colspan="2"><strong>TOTAL</strong></td>'
        f'<td><strong>{passed}/{total} ({total_pct}%)</strong></td><td>—</td></tr>'
        f'</tbody></table>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px;
       background: #f0f2f5; color: #222; line-height: 1.5; }
h1 { font-size: 1.6rem; margin-bottom: .4rem; }
h2 { font-size: 1.2rem; margin: 0 0 .5rem; }
h5 { font-size: .9rem; color: #555; margin-bottom: .4rem; }
small { font-weight: normal; color: #777; }
code { background: #e8e8e8; padding: 1px 4px; border-radius: 3px; font-size: .85em; }

.global-summary { background: #fff; padding: 1.5rem 2rem; border-bottom: 2px solid #ddd; }
.trace-meta { color: #666; margin: .4rem 0 1rem; font-size: .85rem; }

.summary-table { border-collapse: collapse; width: 100%; margin-top: .5rem; }
.summary-table th, .summary-table td { padding: .45rem .7rem; text-align: left;
  border-bottom: 1px solid #e0e0e0; }
.summary-table th { background: #f5f5f5; font-weight: 600; }
.total-row td { background: #f9f9f9; border-top: 2px solid #ccc; }

.model-section { background: #fff; margin: 1.2rem 2rem; padding: 1.2rem 1.5rem;
  border-radius: 8px; border: 1px solid #ddd; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
.model-id { font-size: .78rem; color: #999; }
.model-meta { color: #888; font-size: .82rem; margin: .3rem 0 .8rem; }
.model-rate { margin: .5rem 0 .2rem; display: flex; align-items: center; gap: .8rem; }
.rate-bar-bg { flex: 1; height: 8px; background: #e8e8e8; border-radius: 4px; overflow: hidden; }
.rate-bar { height: 100%; border-radius: 4px; transition: width .3s; }

.prompt-init { margin-bottom: .8rem; }
.prompt-init > summary { cursor: pointer; color: #555; font-size: .85rem;
  padding: .3rem .5rem; background: #f7f7f7; border-radius: 4px; }
.prompt-pre { font-family: 'Courier New', monospace; font-size: .78rem; white-space: pre-wrap;
  background: #fafafa; border: 1px solid #e0e0e0; padding: .8rem; border-radius: 4px;
  max-height: 300px; overflow-y: auto; }

.sheets-container { display: flex; flex-direction: column; gap: .5rem; }
.sheet-details { border: 1px solid #e5e5e5; border-radius: 6px; overflow: hidden; }
.sheet-details > summary { cursor: pointer; padding: .5rem .8rem;
  background: #fafafa; user-select: none; display: flex; align-items: center; gap: .5rem; }
.sheet-details > summary:hover { background: #f0f0f0; }
.sheet-body { padding: .8rem 1rem; display: flex; flex-direction: column; gap: .6rem; }

.attempt-block { border: 1px solid #ebebeb; border-radius: 5px; padding: .6rem .8rem; }
.attempt-grid { display: flex; flex-direction: column; gap: .3rem; margin-top: .3rem; }
.attempt-row { display: grid; grid-template-columns: 100px 1fr; gap: .5rem; align-items: start; }
.row-label { font-weight: 600; color: #555; font-size: .82rem; padding-top: .1rem; }
.row-val { font-size: .85rem; }

.badge { display: inline-block; padding: 1px 8px; border-radius: 10px;
  font-size: .78rem; font-weight: 600; }
.badge-pass { background: #d4edda; color: #155724; }
.badge-fail { background: #f8d7da; color: #721c24; }
.badge-skip { background: #e2e3e5; color: #495057; }

.score-badge { display: inline-flex; align-items: center; justify-content: center;
  width: 22px; height: 22px; border-radius: 50%; font-size: .75rem; font-weight: 700; }
.score-5 { background: #27ae60; color: #fff; }
.score-4 { background: #a8d5b5; color: #155724; }
.score-3 { background: #f39c12; color: #fff; }
.score-low { background: #e74c3c; color: #fff; }

.justif-list { margin: .3rem 0 0 1rem; font-size: .8rem; color: #555; }
.justif-list li { margin-bottom: .1rem; }

.ok   { color: #27ae60; }
.fail { color: #e74c3c; font-weight: 600; }

details > summary { cursor: pointer; }
.diff-block { font-size: .76rem; background: #1e1e1e; color: #d4d4d4;
  padding: .7rem; border-radius: 4px; overflow-x: auto; white-space: pre;
  max-height: 400px; overflow-y: auto; margin-top: .4rem; }
.diff-add  { color: #6db86d; display: block; }
.diff-del  { color: #e06c75; display: block; }
.diff-ctx  { color: #56b6c2; display: block; }
.diff-hdr  { color: #888; display: block; }
.diff-unchanged { color: #d4d4d4; display: block; }
"""


# ─────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────

def render(trace_path: Path, out_path: Path) -> None:
    events = load_events(trace_path)
    data   = build_structure(events)

    # Recover initial prompt from first sheet_prompt files if available
    # (fallback: reconstruct from model_start)
    initial_prompt = "(prompt initial non disponible dans la trace)"
    for ev in events:
        if ev.get("event") == "claude_improve":
            # We know the prompt_after — first attempt is the initial prompt seen by Claude
            # But we don't have the initial explicitly. We'll show it from the first attempt's diff.
            break

    global_html  = render_global(data, trace_path)
    models_html  = "".join(render_model(m, initial_prompt) for m in data["models"])

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Trace — Optimisation de prompt</title>
<style>{CSS}</style>
</head>
<body>
{global_html}
{models_html}
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"✓ HTML généré : {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render trace.jsonl as HTML.")
    p.add_argument(
        "--trace", type=Path,
        default=ROOT_DIR / "data" / "optimized_prompts" / "trace.jsonl",
        help="Path to trace.jsonl",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output HTML path (default: same dir as trace, named trace.html)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trace.exists():
        print(f"Fichier introuvable : {args.trace}", file=sys.stderr)
        sys.exit(1)
    out = args.out or args.trace.with_suffix(".html")
    render(args.trace, out)


if __name__ == "__main__":
    main()
