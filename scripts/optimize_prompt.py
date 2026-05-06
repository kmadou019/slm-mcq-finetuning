#!/usr/bin/env python3
"""
Iterative prompt optimization for MCQ generation via B3 distractor quality feedback.

For each target model, runs a feedback loop:
  generate MCQ → evaluate B3 (GPT-4o) → if fails, Claude improves prompt → repeat
across k LISA sheets, producing a model-specific optimized prompt.

Usage:
  python scripts/optimize_prompt.py --k 20 --max-attempts 5
  python scripts/optimize_prompt.py --models mistral_7b qwen3_0_6b --k 5 --max-attempts 3
"""

import argparse
import ast
import gc
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

load_dotenv(ROOT_DIR / ".env")

from eval.llm_evaluation import call_openai_api


# ─────────────────────────────────────────────────────────────────
# Tracer — structured JSONL events + console output
# ─────────────────────────────────────────────────────────────────

class Tracer:
    """
    Writes one JSON object per line to <output_dir>/trace.jsonl.
    Stream in real time with:  tail -f data/optimized_prompts/trace.jsonl
    """

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._path = output_dir / "trace.jsonl"
        self._fh = self._path.open("a", encoding="utf-8")
        self._t0 = time.monotonic()
        print(f"[tracer] → {self._path}")

    def log(self, event: str, **kwargs) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.monotonic() - self._t0, 2),
            "event": event,
            **kwargs,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    # ── convenience wrappers ──────────────────────────────────────

    def model_start(self, save_name: str, model_id: str, n_sheets: int, max_attempts: int) -> None:
        self.log("model_start", save_name=save_name, model_id=model_id,
                 n_sheets=n_sheets, max_attempts=max_attempts)
        print(f"\n{'='*60}")
        print(f"  [{_ts()}] Modèle : {save_name}")
        print(f"  Fiches : {n_sheets}  |  max_attempts : {max_attempts}")
        print(f"{'='*60}")

    def model_loaded(self, save_name: str, duration_s: float) -> None:
        self.log("model_loaded", save_name=save_name, duration_s=round(duration_s, 1))
        print(f"  [{_ts()}] Pipeline chargé en {duration_s:.1f}s")

    def sheet_start(self, save_name: str, sheet_idx: int, sheet_id, n_sheets: int) -> None:
        self.log("sheet_start", save_name=save_name, sheet_idx=sheet_idx, sheet_id=sheet_id)
        print(f"\n  [{_ts()}] fiche {sheet_idx+1}/{n_sheets}  id={sheet_id}")

    def generation_done(self, save_name: str, sheet_idx: int, attempt: int,
                        duration_s: float, mcq: "MCQQuestion | None") -> None:
        ok = mcq is not None
        self.log("generation", save_name=save_name, sheet_idx=sheet_idx, attempt=attempt,
                 duration_s=round(duration_s, 2), success=ok,
                 question=mcq.question[:120] if ok else None,
                 correct_option=mcq.correct_option if ok else None)
        status = f"ok ({duration_s:.1f}s)" if ok else "INVALIDE"
        print(f"    [{_ts()}] génération : {status}", end="")

    def b3_result(self, save_name: str, sheet_idx: int, attempt: int,
                  duration_s: float, passes: bool, scores: list, justifs: list) -> None:
        self.log("b3_eval", save_name=save_name, sheet_idx=sheet_idx, attempt=attempt,
                 duration_s=round(duration_s, 2), passes=passes,
                 scores=scores, justifs=justifs)
        badge = "✅" if passes else "❌"
        print(f"  →  B3={badge} scores={scores}  ({duration_s:.1f}s)")
        if not passes and justifs:
            for j in justifs:
                print(f"         · {j}")

    def claude_improve(self, save_name: str, sheet_idx: int, attempt: int,
                       duration_s: float, prompt_before: str, prompt_after: str) -> None:
        diff_chars = len(prompt_after) - len(prompt_before)
        self.log("claude_improve", save_name=save_name, sheet_idx=sheet_idx, attempt=attempt,
                 duration_s=round(duration_s, 2),
                 chars_before=len(prompt_before), chars_after=len(prompt_after),
                 chars_diff=diff_chars, prompt_after=prompt_after)
        sign = "+" if diff_chars >= 0 else ""
        print(f"    [{_ts()}] Claude → prompt modifié ({sign}{diff_chars} chars, {duration_s:.1f}s)")

    def sheet_done(self, save_name: str, sheet_idx: int, sheet_id,
                   passed: bool, n_attempts: int) -> None:
        self.log("sheet_done", save_name=save_name, sheet_idx=sheet_idx,
                 sheet_id=sheet_id, passed=passed, n_attempts=n_attempts)
        verdict = "✅ passée" if passed else f"❌ non passée ({n_attempts} tentatives)"
        print(f"    [{_ts()}] fiche {sheet_idx} : {verdict}")

    def model_done(self, save_name: str, duration_s: float,
                   passed: int, total: int) -> None:
        self.log("model_done", save_name=save_name, duration_s=round(duration_s, 1),
                 passed=passed, total=total,
                 pass_rate=round(passed / total, 3) if total else 0)
        print(f"\n  [{_ts()}] ✓ {save_name} : {passed}/{total} "
              f"({100*passed//total if total else 0}%)  durée={duration_s:.0f}s")

    def global_summary(self, per_model: list[dict]) -> None:
        total_sheets  = sum(m["total"]  for m in per_model)
        total_passed  = sum(m["passed"] for m in per_model)
        total_elapsed = round(time.monotonic() - self._t0, 1)
        self.log("global_summary", models=per_model,
                 total_sheets=total_sheets, total_passed=total_passed,
                 total_elapsed_s=total_elapsed)
        print(f"\n{'='*60}")
        print(f"  RÉSUMÉ GLOBAL — {len(per_model)} modèle(s)")
        print(f"{'='*60}")
        for m in per_model:
            rate = 100 * m["passed"] // m["total"] if m["total"] else 0
            print(f"  {m['save_name']:20s}  {m['passed']}/{m['total']}  ({rate}%)")
        print(f"  {'─'*40}")
        rate = 100 * total_passed // total_sheets if total_sheets else 0
        print(f"  {'TOTAL':20s}  {total_passed}/{total_sheets}  ({rate}%)")
        print(f"  Durée totale : {total_elapsed:.0f}s")
        print(f"  Trace        : {self._path}")
        print(f"{'='*60}")


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ─────────────────────────────────────────────────────────────────
# Models targeted for optimization
# ─────────────────────────────────────────────────────────────────

TARGET_MODELS: dict[str, str] = {
    # Large (27B+)
    "medgemma_27b":    "google/medgemma-27b-it",
    "gemma4_31b":      "google/gemma-4-31B-it",
    "mixtral_8x7b":    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "magistral_small": "mistralai/Magistral-Small-2509",
    "nemotron_30b":    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "qwen3_5_35b":     "Qwen/Qwen3.5-35B-A3B",
    # Medium (7-9B)
    "mistral_7b":      "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3_8b":       "meta-llama/Llama-3.1-8B-Instruct",
    "openbiollm_8b":   "antonkirk/Llama3-Instruct-OpenBioLLM-8B-merged",
    "apertus_8b":      "swiss-ai/Apertus-8B-Instruct-2509",
    "gemma2_9b":       "google/gemma-2-9b-it",
    "eurollm_9b":      "utter-project/EuroLLM-9B-Instruct",
    # Small — one only
    "qwen3_0_6b":      "Qwen/Qwen3-0.6B",
}

# Models that require thinking to be disabled
_THINKING_SAVE_NAMES = {"nemotron_30b", "qwen3_5_35b"}


# ─────────────────────────────────────────────────────────────────
# Prompt template (starting point for optimization)
# Inlined from notebooks/generate_mcq.py to avoid ollama import
# ─────────────────────────────────────────────────────────────────

# Sentinel that Claude won't confuse with a Python format variable.
# Using {content} caused Claude to fill it in or drop it during improve_prompt.
_SENTINEL = "<<<CONTENU_EDUCATIF>>>"

_QCM_GUIDELINES = """\
Règles pour la question (stem) :
- La question doit être compréhensible sans lire les propositions
- Rédiger à la forme affirmative — éviter la négation ("laquelle n'est PAS…")
- Ne pas surcharger le stem d'informations non pertinentes à l'objectif évalué

Règles pour les propositions :
- 4 propositions (a, b, c, d), une seule exacte (QRU)
- Propositions homogènes, parallèles, d'un niveau de granularité similaire
- Propositions exprimées à la forme affirmative
- Longueur et précision similaires entre toutes les propositions — la bonne réponse ne doit pas se distinguer par sa longueur
- La bonne réponse ne doit pas reprendre les mots du stem (cluing)
- Distracteurs plausibles mais factuellement incorrects
- Propositions courtes et concises

Propositions INTERDITES :
- "Toutes les propositions précédentes sont correctes"
- "Aucune des propositions précédentes"
- "A et C sont correctes" (combinaison de propositions)
- Toute proposition absurde ou trivialement éliminable

Justifications :
- Fournir une justification pédagogique pour chaque proposition (correcte ou incorrecte)
- Fournir un commentaire global sur ce que la question évalue ou un piège courant"""

_JSON_OUTPUT_BLOCK = """\
CONTRAINTES STRICTES DE SORTIE :
1. La sortie doit être STRICTEMENT un unique objet JSON valide.
2. Interdiction ABSOLUE d'ajouter :
   - des blocs ```json
   - du texte avant ou après le JSON
   - des explications hors champs JSON
3. Le champ "correct_option" doit contenir EXACTEMENT une lettre minuscule parmi : "a", "b", "c", "d".
4. Utiliser uniquement des doubles quotes : "..."
5. Le JSON doit contenir EXACTEMENT les 11 champs suivants :

{
  "question": "...",
  "question_comment": "...",
  "option_a": "...",
  "option_a_comment": "...",
  "option_b": "...",
  "option_b_comment": "...",
  "option_c": "...",
  "option_c_comment": "...",
  "option_d": "...",
  "option_d_comment": "...",
  "correct_option": "a"
}

RÈGLES POUR LES COMMENTAIRES :
- Chaque commentaire d'option doit expliquer brièvement pourquoi l'option est correcte ou incorrecte.
- Le commentaire global de la question doit expliquer ce que la question évalue ou signaler un piège courant.
- Les commentaires doivent être factuels, concis et pédagogiques."""


def _build_initial_prompt() -> str:
    return (
        f"À partir du contenu éducatif suivant, générez exactement une question à choix unique "
        f"avec quatre options de réponse (a, b, c, d), dont une seule est correcte.\n\n"
        f"CONSIGNES DE RÉDACTION :\n"
        f"La question doit évaluer la compréhension des idées principales du contenu fourni.\n\n"
        f"{_QCM_GUIDELINES}\n\n{_JSON_OUTPUT_BLOCK}\n\n"
        f"CONTENU ÉDUCATIF :\n{_SENTINEL}\n\n"
        f"INSTRUCTION FINALE :\nRépondez UNIQUEMENT avec un unique objet JSON valide, sans aucun texte en dehors."
    )


# ─────────────────────────────────────────────────────────────────
# MCQ schema + parsing
# ─────────────────────────────────────────────────────────────────

class MCQQuestion(BaseModel):
    question: str
    question_comment: Optional[str] = ""
    option_a: str
    option_a_comment: Optional[str] = ""
    option_b: str
    option_b_comment: Optional[str] = ""
    option_c: str
    option_c_comment: Optional[str] = ""
    option_d: str
    option_d_comment: Optional[str] = ""
    correct_option: str


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    snippet = text[start : end + 1]
    return json.dumps(ast.literal_eval(snippet), ensure_ascii=False)


def _validate_mcq(raw: str) -> MCQQuestion | None:
    try:
        return MCQQuestion.model_validate_json(raw)
    except (ValidationError, Exception) as e:
        print(f"    [parse] {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# Generation (HuggingFace pipeline)
# ─────────────────────────────────────────────────────────────────

def _load_pipeline(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import pipeline as hf_pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    try:
        pipe = hf_pipeline(
            "text-generation", model=model_id, tokenizer=tokenizer,
            device_map="cuda", dtype="bfloat16",
        )
    except (ValueError, OSError):
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda", torch_dtype="auto", trust_remote_code=True
        )
        pipe = hf_pipeline("text-generation", model=model_obj, tokenizer=tokenizer)
    return pipe


def _unload_pipeline(pipe) -> None:
    import torch
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def generate_with_template(
    content: str, prompt_template: str, pipe, save_name: str, temperature: float = 0.7
) -> MCQQuestion | None:
    full_prompt = prompt_template.replace(_SENTINEL, content)
    messages = [{"role": "user", "content": full_prompt}]
    kwargs: dict = dict(
        max_new_tokens=2048, temperature=temperature, top_p=1.0,
        do_sample=True, return_full_text=False,
    )
    if save_name in _THINKING_SAVE_NAMES:
        kwargs["tokenize_kwargs"] = {"enable_thinking": False}
    try:
        raw = pipe(messages, **kwargs)[0]["generated_text"]
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return _validate_mcq(_extract_json(raw))
    except Exception as e:
        print(f"    [gen] {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# B3 evaluation (GPT-4o)
# ─────────────────────────────────────────────────────────────────

def _load_b3_prompt() -> str:
    with open(ROOT_DIR / "src" / "eval" / "prompts.json") as f:
        return json.load(f)["distractors_quality_prompt"]


def evaluate_b3(
    mcq: MCQQuestion, content: str, client: OpenAI, system_prompt: str
) -> dict:
    opts = f"a) {mcq.option_a}\nb) {mcq.option_b}\nc) {mcq.option_c}\nd) {mcq.option_d}"
    user_prompt = (
        f"Context:\n-----\n{content}\n-----\n"
        f"-----\nQuestion:\n{mcq.question}\nOptions:\n{opts}\n"
        f"Correct option: {mcq.correct_option}\n-----"
    )
    time.sleep(0.4)  # GPT-4o rate limit
    raw = call_openai_api(client, system_prompt, user_prompt, temp=0.0, max_completion_tokens=4000)
    if not raw:
        return {"passes": False, "scores": [], "justifs": [], "raw": ""}
    try:
        clean = re.sub(r"```(?:json)?\s*", "", raw).strip()
        data = json.loads(clean)
        scores = [int(d["score"]) for d in data]
        justifs = [d.get("justif", "") for d in data]
        passes = sum(s >= 4 for s in scores) >= 2
    except Exception as e:
        print(f"    [b3] parse error: {e}")
        passes, scores, justifs = False, [], []
    return {"passes": passes, "scores": scores, "justifs": justifs, "raw": raw}


# ─────────────────────────────────────────────────────────────────
# Claude prompt improvement
# ─────────────────────────────────────────────────────────────────

_IMPROVE_SYSTEM = (
    "Tu es un expert en ingénierie de prompts pour la génération de QCM médicaux en français.\n"
    "Tu reçois un prompt de génération et un exemple de QCM dont les distracteurs ont été "
    "jugés de mauvaise qualité (score GPT-4o < 4/5).\n\n"
    "DÉMARCHE EN DEUX TEMPS :\n\n"
    "1. DIAGNOSTIC — Identifie la cause racine dans le prompt lui-même :\n"
    "   Pourquoi le prompt pousse-t-il le modèle à produire ce type d'erreur ? "
    "   Est-ce une règle absente, une instruction ambiguë, un manque de structure cognitive, "
    "   un exemple contre-productif ? Ne te contente pas de corriger le symptôme visible.\n\n"
    "2. GÉNÉRALISATION — Tire une leçon transférable :\n"
    "   La modification que tu apportes doit améliorer la génération sur N'IMPORTE QUELLE "
    "   fiche LISA médicale future, pas seulement sur le contenu de cet exemple. "
    "   Pense à ce que le modèle doit faire différemment structurellement.\n\n"
    "CONTRAINTES ABSOLUES :\n"
    f"- Conserver la balise {_SENTINEL} exactement à sa place — c'est un marqueur technique, "
    "ne le modifie pas, ne le supprime pas, ne le commente pas\n"
    "- Conserver les CONTRAINTES STRICTES DE SORTIE (format JSON) inchangées\n"
    "- Retourner UNIQUEMENT le prompt complet modifié, sans explication ni commentaire\n"
    "- Ne jamais injecter de contenu spécifique à l'exemple fourni (valeurs chiffrées, "
    "pathologies, médicaments, mécanismes précis) dans les règles du prompt. "
    "Les règles doivent être génériques. Si tu illustres une règle par un exemple, "
    "utilise des placeholders abstraits : '[valeur correcte]', '[contre-indication]', "
    "'[mécanisme principal]' — jamais du contenu issu de la fiche évaluée."
)


def improve_prompt(
    prompt_template: str,
    mcq: MCQQuestion,
    b3: dict,
    content: str,
    model_name: str,
) -> str:
    distractors = [o for o in ("a", "b", "c", "d") if o != mcq.correct_option]
    score_lines = ""
    for opt, score, justif in zip(distractors, b3.get("scores", []), b3.get("justifs", [])):
        score_lines += f"  - Option {opt}) « {getattr(mcq, f'option_{opt}')} » → {score}/5 — {justif}\n"

    full_prompt = (
        f"{_IMPROVE_SYSTEM}\n\n"
        f"Modèle cible : {model_name}\n\n"
        f"Prompt actuel :\n{prompt_template}\n\n"
        f"QCM ayant échoué (distracteurs de mauvaise qualité) :\n"
        f"Question : {mcq.question}\n"
        f"Réponse correcte ({mcq.correct_option}) : {getattr(mcq, f'option_{mcq.correct_option}')}\n"
        f"Scores distracteurs :\n{score_lines}"
        f"Extrait du contenu source :\n{content[:600]}\n\n"
        f"→ Améliore le prompt pour que ce modèle génère de meilleurs distracteurs."
    )
    result = subprocess.run(
        ["claude", "-p", full_prompt],
        capture_output=True, text=True, timeout=600,
    )
    raw = result.stdout.strip()
    if not raw:
        print(f"    [Claude] ⚠ réponse vide (stderr: {result.stderr[:100]}), prompt conservé")
        return prompt_template

    # Extract prompt from fenced code block if Claude wrapped it
    import re
    code_block = re.search(r"```(?:\w*\n)?(.*?)```", raw, re.DOTALL)
    improved = code_block.group(1).strip() if code_block else raw

    if _SENTINEL not in improved:
        print(f"    [Claude] ⚠ balise {_SENTINEL} absente, prompt conservé")
        return prompt_template
    return improved


# ─────────────────────────────────────────────────────────────────
# Optimization loop (one model)
# ─────────────────────────────────────────────────────────────────

def optimize_model(
    save_name: str,
    model_id: str,
    sheets: list[dict],
    initial_prompt: str,
    openai_client: OpenAI,
    b3_system_prompt: str,
    max_attempts: int,
    output_dir: Path,
    tracer: Tracer,
) -> str:
    model_dir = output_dir / save_name
    model_dir.mkdir(parents=True, exist_ok=True)

    tracer.model_start(save_name, model_id, len(sheets), max_attempts)

    t_load = time.monotonic()
    pipe = _load_pipeline(model_id)
    tracer.model_loaded(save_name, time.monotonic() - t_load)

    prompt = initial_prompt
    log: list[dict] = []
    t_model = time.monotonic()

    try:
        for sheet_idx, sheet in enumerate(sheets):
            content: str = sheet["content_raw"]
            sheet_id = sheet.get("id", sheet_idx)
            tracer.sheet_start(save_name, sheet_idx, sheet_id, len(sheets))
            passed = False
            attempt = 0

            for attempt in range(max_attempts):
                print(f"    [{_ts()}] tentative {attempt+1}/{max_attempts}", end="", flush=True)

                t_gen = time.monotonic()
                mcq = generate_with_template(content, prompt, pipe, save_name)
                tracer.generation_done(save_name, sheet_idx, attempt,
                                       time.monotonic() - t_gen, mcq)
                if mcq is None:
                    print()  # newline after inline status
                    break

                t_b3 = time.monotonic()
                b3 = evaluate_b3(mcq, content, openai_client, b3_system_prompt)
                tracer.b3_result(save_name, sheet_idx, attempt,
                                 time.monotonic() - t_b3,
                                 b3["passes"], b3["scores"], b3["justifs"])

                if b3["passes"]:
                    passed = True
                    break

                if attempt < max_attempts - 1:
                    prev_prompt = prompt
                    t_claude = time.monotonic()
                    prompt = improve_prompt(prompt, mcq, b3, content, model_id)
                    tracer.claude_improve(save_name, sheet_idx, attempt,
                                          time.monotonic() - t_claude,
                                          prev_prompt, prompt)

            tracer.sheet_done(save_name, sheet_idx, sheet_id, passed, attempt + 1)
            log.append({
                "sheet_idx": sheet_idx, "sheet_id": sheet_id,
                "attempts": attempt + 1, "passed": passed,
            })

    finally:
        _unload_pipeline(pipe)

    (model_dir / "final_prompt.txt").write_text(prompt, encoding="utf-8")
    (model_dir / "optimization_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    passed_count = sum(1 for e in log if e["passed"])
    tracer.model_done(save_name, time.monotonic() - t_model, passed_count, len(sheets))
    return prompt


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimize MCQ generation prompt per model via B3 feedback loop."
    )
    p.add_argument(
        "--models", nargs="+", choices=list(TARGET_MODELS), default=list(TARGET_MODELS),
        metavar="MODEL",
        help=f"Models to optimize. Available: {list(TARGET_MODELS.keys())}",
    )
    p.add_argument("--k", type=int, default=20, help="Number of LISA sheets (default: 20)")
    p.add_argument(
        "--max-attempts", type=int, default=5, dest="max_attempts",
        help="Max attempts per sheet before moving on (default: 5)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=ROOT_DIR / "data" / "optimized_prompts",
        dest="output_dir", help="Output directory for optimized prompts",
    )
    p.add_argument(
        "--lisa-csv", type=Path, default=ROOT_DIR / "data" / "lisa_sheets.csv",
        dest="lisa_csv", help="Path to LISA sheets CSV",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for sheet sampling (default: 42)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    openai_client = OpenAI(api_key=openai_key)
    b3_prompt = _load_b3_prompt()

    df = pd.read_csv(args.lisa_csv)
    # One sheet per parent item (IC-XXX) for thematic diversity
    df["_item_parent"] = df["id"].str.extract(r"^([A-Z]+-\d+)")
    one_per_parent = (
        df.groupby("_item_parent", group_keys=False)
        .apply(lambda g: g.sample(1, random_state=args.seed))
        .reset_index(drop=True)
    )
    sheets = one_per_parent.sample(
        n=min(args.k, len(one_per_parent)), random_state=args.seed
    ).drop(columns=["_item_parent"], errors="ignore").to_dict("records")
    print(f"✓ {len(sheets)} fiches LISA chargées (1 par item parent) depuis {args.lisa_csv}")

    initial_prompt = _build_initial_prompt()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tracer = Tracer(args.output_dir)
    per_model: list[dict] = []

    failed_models: list[str] = []

    try:
        for save_name in args.models:
            final_prompt_path = args.output_dir / save_name / "final_prompt.txt"
            if final_prompt_path.exists():
                print(f"\n  [{_ts()}] ⏭  {save_name} — déjà terminé, skip")
                tracer.log("model_skipped", save_name=save_name,
                           reason="final_prompt.txt already exists")
                log_path = args.output_dir / save_name / "optimization_log.json"
                if log_path.exists():
                    log = json.loads(log_path.read_text())
                    per_model.append({
                        "save_name": save_name,
                        "passed": sum(1 for e in log if e["passed"]),
                        "total": len(log),
                    })
                continue

            try:
                optimize_model(
                    save_name=save_name,
                    model_id=TARGET_MODELS[save_name],
                    sheets=sheets,
                    initial_prompt=initial_prompt,
                    openai_client=openai_client,
                    b3_system_prompt=b3_prompt,
                    max_attempts=args.max_attempts,
                    output_dir=args.output_dir,
                    tracer=tracer,
                )
                log = json.loads(
                    (args.output_dir / save_name / "optimization_log.json").read_text()
                )
                per_model.append({
                    "save_name": save_name,
                    "passed": sum(1 for e in log if e["passed"]),
                    "total": len(log),
                })
            except Exception as exc:
                failed_models.append(save_name)
                tracer.log("model_error", save_name=save_name, error=str(exc))
                print(f"\n  [{_ts()}] ✗ {save_name} — ERREUR : {exc}")
                print(f"  Passage au modèle suivant...")

        tracer.global_summary(per_model)

        if failed_models:
            print(f"\n  ⚠ Modèles en erreur : {', '.join(failed_models)}")

    finally:
        tracer.close()
        from render_trace import render
        render(args.output_dir / "trace.jsonl", args.output_dir / "trace.html")


if __name__ == "__main__":
    main()
