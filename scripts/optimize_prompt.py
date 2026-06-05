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
import random
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
from eval.question_check import is_question
from eval.negation import starts_with_negation
from eval.originality import calculate_originality_for_df
from eval.readability import calculate_readability_for_df
from eval.relevance import calculate_relevance_for_df


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
                       duration_s: float, prompt_before: str, prompt_after: str,
                       stage: str = "") -> None:
        diff_chars = len(prompt_after) - len(prompt_before)
        self.log("claude_improve", save_name=save_name, sheet_idx=sheet_idx, attempt=attempt,
                 duration_s=round(duration_s, 2), stage=stage,
                 chars_before=len(prompt_before), chars_after=len(prompt_after),
                 chars_diff=diff_chars, prompt_after=prompt_after)
        sign = "+" if diff_chars >= 0 else ""
        stage_label = f" [{stage}]" if stage else ""
        print(f"    [{_ts()}] Claude{stage_label} → prompt modifié ({sign}{diff_chars} chars, {duration_s:.1f}s)")

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
    # Small
    "qwen3_0_6b":      "Qwen/Qwen3-0.6B",                                    # 0.6B
    # Medium (7–9B)
    "mistral_7b":      "mistralai/Mistral-7B-Instruct-v0.3",                  # 7B
    "llama3_8b":       "meta-llama/Llama-3.1-8B-Instruct",                    # 8B
    "openbiollm_8b":   "antonkirk/Llama3-Instruct-OpenBioLLM-8B-merged",      # 8B
    "apertus_8b":      "swiss-ai/Apertus-8B-Instruct-2509",                   # 8B
    "gemma2_9b":       "google/gemma-2-9b-it",                                # 9B
    "eurollm_9b":      "utter-project/EuroLLM-9B-Instruct",                   # 9B
    # Large (24B+)
    "magistral_small": "mistralai/Magistral-Small-2509",                       # ~24B
    "medgemma_27b":    "google/medgemma-27b-it",                               # 27B
    "nemotron_30b":    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",          # 30B total (3B active, MoE)
    "gemma4_31b":      "google/gemma-4-31B-it",                                # 31B
    "qwen3_5_35b":     "Qwen/Qwen3.5-35B-A3B",                                # 35B total (3B active, MoE)
    "mixtral_8x7b":    "mistralai/Mixtral-8x7B-Instruct-v0.1",                # ~47B total (14B active, MoE)
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

# Sentinels for the two-stage pipeline (stage 2: distractors)
_SENTINEL_Q   = "<<<QUESTION>>>"
_SENTINEL_ANS = "<<<REPONSE_CORRECTE>>>"

# Delimiters Claude must wrap its output in, so we can extract the prompt cleanly.
_PROMPT_START = "<<<PROMPT_DEBUT>>>"
_PROMPT_END   = "<<<PROMPT_FIN>>>"

# Section markers: allow extracting and merging question/distractor rules
_SECTION_Q_START = "<<<DEBUT_REGLES_QUESTION>>>"
_SECTION_Q_END   = "<<<FIN_REGLES_QUESTION>>>"
_SECTION_D_START = "<<<DEBUT_REGLES_DISTRACTEURS>>>"
_SECTION_D_END   = "<<<FIN_REGLES_DISTRACTEURS>>>"

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

# Distractor-only guidelines (no question-stem rules) — used in stage 2 prompt
_DISTRACTOR_GUIDELINES = """\
Règles pour les distracteurs :
- Homogènes, parallèles à la bonne réponse, d'un niveau de granularité similaire
- Exprimés à la forme affirmative
- Longueur et précision similaires à la bonne réponse — ne pas se distinguer par la longueur
- Plausibles mais factuellement incorrects, ancrés dans le contenu éducatif
- Concis

Distracteurs INTERDITS :
- "Toutes les propositions précédentes sont correctes"
- "Aucune des propositions précédentes"
- Toute combinaison de propositions ("A et C sont correctes")
- Toute proposition absurde ou trivialement éliminable

Commentaires :
- Fournir pour chaque distracteur une explication concise de pourquoi il est incorrect"""

# JSON output block for stage 1 (question + correct answer only)
_JSON_QUESTION_BLOCK = """\
CONTRAINTES STRICTES DE SORTIE :
1. La sortie doit être STRICTEMENT un unique objet JSON valide.
2. Interdiction ABSOLUE d'ajouter :
   - des blocs ```json
   - du texte avant ou après le JSON
   - des explications hors champs JSON
3. Utiliser uniquement des doubles quotes : "..."
4. Le JSON doit contenir EXACTEMENT les 3 champs suivants :

{
  "question": "...",
  "question_comment": "...",
  "correct_answer": "..."
}

RÈGLES POUR LES CHAMPS :
- "question" : forme interrogative directe, se terminant par "?", sans négation
- "question_comment" : ce que la question évalue ou un piège pédagogique courant
- "correct_answer" : réponse correcte concise, factuelle, ne reprenant pas les mots de la question"""

# JSON output block for stage 2 (3 distractors only)
_JSON_DISTRACTOR_BLOCK = """\
CONTRAINTES STRICTES DE SORTIE :
1. La sortie doit être STRICTEMENT un unique objet JSON valide.
2. Interdiction ABSOLUE d'ajouter :
   - des blocs ```json
   - du texte avant ou après le JSON
   - des explications hors champs JSON
3. Utiliser uniquement des doubles quotes : "..."
4. Le JSON doit contenir EXACTEMENT les 6 champs suivants :

{
  "distractor_1": "...",
  "distractor_1_comment": "...",
  "distractor_2": "...",
  "distractor_2_comment": "...",
  "distractor_3": "...",
  "distractor_3_comment": "..."
}

RÈGLES POUR LES COMMENTAIRES :
- Chaque commentaire doit expliquer brièvement pourquoi ce distracteur est incorrect.
- Les commentaires doivent être factuels, concis et pédagogiques."""

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


def _build_initial_question_prompt() -> str:
    """Stage 1 prompt: generates question + correct answer only."""
    return (
        f"À partir du contenu éducatif suivant, générez une question évaluative et sa réponse correcte.\n\n"
        f"CONSIGNES DE RÉDACTION :\n"
        f"La question doit évaluer la compréhension des idées principales du contenu fourni.\n\n"
        f"{_SECTION_Q_START}\n"
        f"Règles pour la question (stem) :\n"
        f"- Rédiger à la forme interrogative directe (se terminer par \"?\")\n"
        f"- Rédiger à la forme affirmative — éviter la négation (\"laquelle n'est PAS…\")\n"
        f"- La question doit être compréhensible sans contexte supplémentaire\n"
        f"- Ne pas paraphraser directement le contenu — évaluer la compréhension, pas la mémorisation\n"
        f"- Ne pas surcharger le stem d'informations non pertinentes à l'objectif évalué\n\n"
        f"Règles pour la réponse correcte :\n"
        f"- Courte et concise (une phrase maximum)\n"
        f"- Factuelle, ancrée dans le contenu fourni\n"
        f"- Ne pas reprendre les mots exacts de la question (cluing)\n"
        f"{_SECTION_Q_END}\n\n"
        f"{_JSON_QUESTION_BLOCK}\n\n"
        f"CONTENU ÉDUCATIF :\n{_SENTINEL}\n\n"
        f"INSTRUCTION FINALE :\nRépondez UNIQUEMENT avec un unique objet JSON valide, sans aucun texte en dehors."
    )


def _build_initial_distractor_prompt() -> str:
    """Stage 2 prompt: generates 3 distractors for a frozen question + correct answer."""
    return (
        f"À partir du contenu éducatif et de la question ci-dessous, générez exactement 3 distracteurs plausibles.\n\n"
        f"CONTEXTE (NE PAS MODIFIER) :\n"
        f"Question : {_SENTINEL_Q}\n"
        f"Réponse correcte : {_SENTINEL_ANS}\n\n"
        f"{_SECTION_D_START}\n"
        f"{_DISTRACTOR_GUIDELINES}\n"
        f"{_SECTION_D_END}\n\n"
        f"{_JSON_DISTRACTOR_BLOCK}\n\n"
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


# Stage 1: question + correct answer only
class MCQStep1(BaseModel):
    question: str
    question_comment: Optional[str] = ""
    correct_answer: str


# Stage 2: 3 distractors only (question frozen from stage 1)
class MCQStep2(BaseModel):
    distractor_1: str
    distractor_1_comment: Optional[str] = ""
    distractor_2: str
    distractor_2_comment: Optional[str] = ""
    distractor_3: str
    distractor_3_comment: Optional[str] = ""


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


def _validate_step1(raw: str) -> MCQStep1 | None:
    try:
        return MCQStep1.model_validate_json(raw)
    except (ValidationError, Exception) as e:
        print(f"    [parse-q] {e}")
        return None


def _validate_step2(raw: str) -> MCQStep2 | None:
    try:
        return MCQStep2.model_validate_json(raw)
    except (ValidationError, Exception) as e:
        print(f"    [parse-d] {e}")
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


def _run_pipeline(full_prompt: str, pipe, save_name: str, temperature: float = 0.7) -> str | None:
    """Shared HF pipeline call used by both stage-1 and stage-2 generators."""
    messages = [{"role": "user", "content": full_prompt}]
    kwargs: dict = dict(
        max_new_tokens=2048, temperature=temperature, top_p=1.0,
        do_sample=True, return_full_text=False,
    )
    if save_name in _THINKING_SAVE_NAMES:
        kwargs["tokenize_kwargs"] = {"enable_thinking": False}
    try:
        raw = pipe(messages, **kwargs)[0]["generated_text"]
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    except Exception as e:
        print(f"    [gen] {e}")
        return None


def generate_question(
    content: str, prompt_q: str, pipe, save_name: str, temperature: float = 0.7
) -> MCQStep1 | None:
    """Stage 1: generate question + correct answer from LISA content."""
    full_prompt = prompt_q.replace(_SENTINEL, content)
    raw = _run_pipeline(full_prompt, pipe, save_name, temperature)
    if raw is None:
        return None
    try:
        return _validate_step1(_extract_json(raw))
    except Exception as e:
        print(f"    [gen-q] {e}")
        return None


def generate_distractors(
    content: str,
    question: str,
    correct_answer: str,
    prompt_d: str,
    pipe,
    save_name: str,
    temperature: float = 0.7,
) -> MCQStep2 | None:
    """Stage 2: generate 3 distractors for a frozen question + correct answer."""
    full_prompt = (
        prompt_d
        .replace(_SENTINEL, content)
        .replace(_SENTINEL_Q, question)
        .replace(_SENTINEL_ANS, correct_answer)
    )
    raw = _run_pipeline(full_prompt, pipe, save_name, temperature)
    if raw is None:
        return None
    try:
        return _validate_step2(_extract_json(raw))
    except Exception as e:
        print(f"    [gen-d] {e}")
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
# Question quality evaluation (local metrics A1, A2, A3, A4, B1)
# ─────────────────────────────────────────────────────────────────

def evaluate_question_quality(question: str, content: str) -> dict:
    """Evaluate question-only metrics locally (no GPT-4o call).

    Seuils : A1=True, A2=True (no negation), A3≥0.75, A4≥12 (FK), B1≥0.5.
    Note: starts_with_negation() uses English patterns only — A2 will always
    pass on French questions. Known bug in the metric, not fixed here.
    """
    # A1, A2 : direct string functions
    a1 = is_question(question)
    a2 = not starts_with_negation(question)

    # A3, A4, B1 : single-row DataFrame API
    df = pd.DataFrame([{"question": question, "content_raw": content}])
    df = calculate_originality_for_df(df, "originality", "question", "content_raw")
    df = calculate_readability_for_df(df, "readability", "question")
    df = calculate_relevance_for_df(df, "relevance", "question", "content_raw")

    a3 = float(df["originality"].iloc[0])
    a4_raw = df["readability"].iloc[0]
    a4 = float(a4_raw) if a4_raw is not None else None
    b1 = float(df["relevance"].iloc[0])

    passes = a1 and a2 and a3 >= 0.75 and (a4 is not None and a4 >= 12) and b1 >= 0.5
    return {"passes": passes, "a1": a1, "a2": a2, "a3": a3, "a4": a4, "b1": b1}


# ─────────────────────────────────────────────────────────────────
# MCQ assembly: merge step1 + step2 → MCQQuestion (shuffled slots)
# ─────────────────────────────────────────────────────────────────

def assemble_mcq(step1: MCQStep1, step2: MCQStep2) -> tuple["MCQQuestion", list[str]]:
    """Randomly assign correct answer + 3 distractors to option slots a-d.

    Returns (MCQQuestion, d_slots) where d_slots[i] is the slot letter ('a'–'d')
    assigned to distractor_{i+1}.  Needed to map GPT-4o scores back to distractors.
    """
    slots = ["a", "b", "c", "d"]
    random.shuffle(slots)
    correct_slot = slots[0]
    d_slots = slots[1:]
    options: dict[str, tuple[str, str]] = {
        correct_slot: (step1.correct_answer, ""),
        d_slots[0]: (step2.distractor_1, step2.distractor_1_comment or ""),
        d_slots[1]: (step2.distractor_2, step2.distractor_2_comment or ""),
        d_slots[2]: (step2.distractor_3, step2.distractor_3_comment or ""),
    }
    return MCQQuestion(
        question=step1.question,
        question_comment=step1.question_comment or "",
        option_a=options["a"][0], option_a_comment=options["a"][1],
        option_b=options["b"][0], option_b_comment=options["b"][1],
        option_c=options["c"][0], option_c_comment=options["c"][1],
        option_d=options["d"][0], option_d_comment=options["d"][1],
        correct_option=correct_slot,
    ), d_slots


# ─────────────────────────────────────────────────────────────────
# Final prompt merge: combine optimized question + distractor rules
# ─────────────────────────────────────────────────────────────────

def _extract_section(text: str, start_tag: str, end_tag: str) -> str:
    m = re.search(re.escape(start_tag) + r"(.*?)" + re.escape(end_tag), text, re.DOTALL)
    if not m:
        raise ValueError(f"Section marker {start_tag!r} not found in prompt")
    return m.group(1).strip()


def build_final_prompt(prompt_q: str, prompt_d: str) -> str:
    """Merge the two optimized stage prompts into a single single-pass prompt (11 fields)."""
    q_rules = _extract_section(prompt_q, _SECTION_Q_START, _SECTION_Q_END)
    d_rules = _extract_section(prompt_d, _SECTION_D_START, _SECTION_D_END)
    return (
        f"À partir du contenu éducatif suivant, générez exactement une question à choix unique "
        f"avec quatre options de réponse (a, b, c, d), dont une seule est correcte.\n\n"
        f"CONSIGNES DE RÉDACTION :\n"
        f"La question doit évaluer la compréhension des idées principales du contenu fourni.\n\n"
        f"{_SECTION_Q_START}\n{q_rules}\n{_SECTION_Q_END}\n\n"
        f"{_SECTION_D_START}\n{d_rules}\n{_SECTION_D_END}\n\n"
        f"{_JSON_OUTPUT_BLOCK}\n\n"
        f"CONTENU ÉDUCATIF :\n{_SENTINEL}\n\n"
        f"INSTRUCTION FINALE :\nRépondez UNIQUEMENT avec un unique objet JSON valide, sans aucun texte en dehors."
    )


# ─────────────────────────────────────────────────────────────────
# Prompt improvement (Qwen via OpenRouter)
# ─────────────────────────────────────────────────────────────────

_IMPROVE_SYSTEM_Q = (
    "Tu es un expert en ingénierie de prompts pour la génération de QCM médicaux en français.\n"
    "Tu reçois : (1) un prompt de génération de question, (2) la question produite, "
    "(3) les critères échoués avec leurs valeurs observées.\n\n"
    "ANALYSE OBLIGATOIRE — avant toute modification, complète ces trois étapes :\n\n"
    "A. LOCALISATION : Pour chaque critère échoué listé dans le message, cite la phrase ou règle "
    "exacte dans le prompt actuel (entre guillemets) qui était censée garantir ce critère — "
    "ou constate son absence complète.\n\n"
    "B. CHAÎNE CAUSALE : Formule pour chaque échec : "
    "«La règle [X] a conduit le modèle à [comportement observé], ce qui produit [échec métrique].» "
    "ou «L'absence de règle sur [X] permet au modèle de [comportement indésirable].»\n\n"
    "C. CORRECTION CIBLÉE : Identifie si la cause est :\n"
    "   - règle absente → l'ajouter\n"
    "   - règle ambiguë → la rendre plus précise avec un exemple positif/négatif\n"
    "   - règle contre-productive → la supprimer ou l'inverser\n"
    "Ne traite que les règles impliquées dans un échec réel. "
    "Ne modifie pas ce qui fonctionne.\n\n"
    "GÉNÉRALISATION : La modification doit améliorer la génération sur N'IMPORTE QUELLE "
    "fiche LISA médicale future. Ne jamais injecter de contenu spécifique à l'exemple "
    "(pathologies, valeurs, mécanismes) — utiliser des placeholders abstraits si nécessaire.\n\n"
    "CONTRAINTES ABSOLUES :\n"
    f"- Conserver la balise {_SENTINEL} exactement à sa place\n"
    f"- Conserver les balises de section {_SECTION_Q_START} et {_SECTION_Q_END} exactement à leur place\n"
    "- Conserver les CONTRAINTES STRICTES DE SORTIE (format JSON) inchangées\n"
    f"- Modifier UNIQUEMENT la section entre {_SECTION_Q_START} et {_SECTION_Q_END}. "
    "Toutes les autres sections doivent rester strictement identiques, mot pour mot.\n"
    f"- Encadrer le prompt complet modifié EXACTEMENT entre {_PROMPT_START} et {_PROMPT_END} "
    f"— aucun texte en dehors. Format attendu :\n{_PROMPT_START}\n[prompt modifié]\n{_PROMPT_END}"
)

_IMPROVE_SYSTEM_D = (
    "Tu es un expert en ingénierie de prompts pour la génération de QCM médicaux en français.\n"
    "Tu reçois : (1) un prompt de génération de distracteurs, (2) la question (gelée) et la "
    "réponse correcte, (3) chaque distracteur produit avec son score GPT-4o (1–5) et la "
    "justification de l'évaluateur.\n\n"
    "ANALYSE OBLIGATOIRE — avant toute modification, complète ces trois étapes :\n\n"
    "A. CLASSIFICATION DES DÉFAUTS : Pour chaque distracteur avec score < 4, identifie sa "
    "catégorie de défaut parmi :\n"
    "   - TROP ÉVIDENT : éliminable sans connaissance médicale (score 1–2)\n"
    "   - TROP PROCHE : presque synonyme de la bonne réponse, risque d'ambiguïté (score 2–3)\n"
    "   - NON PLAUSIBLE : incohérent médicalement pour un étudiant averti (score 1–2)\n"
    "   - HORS SUJET : sans rapport avec la question ou le contenu (score 1–2)\n"
    "   Si plusieurs distracteurs partagent la même catégorie → problème systémique.\n\n"
    "B. LOCALISATION : Pour chaque catégorie de défaut identifiée, cite la phrase ou règle "
    "exacte dans le prompt actuel (entre guillemets) qui était censée l'empêcher — "
    "ou constate son absence. S'appuie sur la justification GPT-4o pour comprendre pourquoi "
    "ce distracteur spécifique a échoué.\n\n"
    "C. CORRECTION CIBLÉE : Identifie si la cause est :\n"
    "   - règle absente → l'ajouter (ex : exiger que chaque distracteur soit discriminant "
    "pour un étudiant de rang B)\n"
    "   - règle trop vague → la préciser avec un critère mesurable ou un exemple négatif\n"
    "   - règle contre-productive → la supprimer ou la reformuler\n"
    "Ne modifie que ce qui est impliqué dans un défaut réel. "
    "Ne touche pas aux règles qui ont produit les bons distracteurs (score ≥ 4).\n\n"
    "GÉNÉRALISATION : La modification doit améliorer la génération sur N'IMPORTE QUELLE "
    "fiche LISA médicale future. Ne jamais injecter de contenu spécifique à l'exemple "
    "(pathologies, valeurs chiffrées, médicaments, mécanismes précis) — "
    "utiliser des placeholders abstraits : '[valeur correcte]', '[contre-indication]'.\n\n"
    "CONTRAINTES ABSOLUES :\n"
    f"- Conserver les trois balises {_SENTINEL}, {_SENTINEL_Q}, {_SENTINEL_ANS} exactement à leur place\n"
    f"- Conserver les balises de section {_SECTION_D_START} et {_SECTION_D_END} exactement à leur place\n"
    "- Conserver les CONTRAINTES STRICTES DE SORTIE (format JSON) inchangées\n"
    f"- Modifier UNIQUEMENT la section entre {_SECTION_D_START} et {_SECTION_D_END}. "
    "Toutes les autres sections doivent rester strictement identiques, mot pour mot.\n"
    f"- Encadrer le prompt complet modifié EXACTEMENT entre {_PROMPT_START} et {_PROMPT_END} "
    f"— aucun texte en dehors. Format attendu :\n{_PROMPT_START}\n[prompt modifié]\n{_PROMPT_END}"
)

def claude(full_prompt: str):
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    response = client.chat.completions.create(
        model="qwen/qwen3.5-397b-a17b",
        messages=[{"role": "user", "content": full_prompt}],
        extra_body={"reasoning": {"enabled": True}},
    )
    return response.choices[0].message



def _extract_improved_prompt(raw: str, fallback: str, required_sentinels: list[str]) -> str | None:
    """Extract and validate the improved prompt from Qwen's response.

    Returns None if extraction fails, any required sentinel is missing,
    or the result is more than 30% shorter than the original.
    """
    m = re.search(re.escape(_PROMPT_START) + r"(.*?)" + re.escape(_PROMPT_END), raw, re.DOTALL)
    if m:
        improved = m.group(1).strip()
    else:
        code_block = re.search(r"```(?:\w*\n)?(.*?)```", raw, re.DOTALL)
        if code_block:
            improved = code_block.group(1).strip()
        else:
            improved = raw
            print(f"    [Qwen] ⚠ balises {_PROMPT_START}/{_PROMPT_END} absentes, extraction brute")

    for sentinel in required_sentinels:
        if sentinel not in improved:
            print(f"    [Qwen] ⚠ balise {sentinel!r} absente, prompt conservé")
            return None

    ratio = len(improved) / len(fallback)
    if ratio < 0.7:
        print(
            f"    [Qwen] ⚠ réduction trop agressive ({len(fallback)}→{len(improved)} chars, "
            f"{100*(1-ratio):.0f}% supprimé) — prompt conservé"
        )
        return None

    return improved


def improve_question_prompt(
    prompt_q: str,
    step1: MCQStep1,
    quality_result: dict,
    model_name: str,
) -> str:
    """Ask Qwen to improve the question-generation section of prompt_q."""
    failures = []
    if not quality_result["a1"]:
        failures.append("A1=False (la question n'est pas sous forme interrogative)")
    if not quality_result["a2"]:
        failures.append("A2=False (négation détectée)")
    a3 = quality_result["a3"]
    if a3 < 0.75:
        failures.append(f"A3={a3:.2f} < 0.75 (originalité insuffisante, trop proche du contenu LISA)")
    a4 = quality_result["a4"]
    if a4 is None or a4 < 12:
        failures.append(f"A4={a4} < 12 (lisibilité FK insuffisante)")
    b1 = quality_result["b1"]
    if b1 < 0.5:
        failures.append(f"B1={b1:.2f} < 0.5 (pertinence cosinus insuffisante)")

    len_before = len(prompt_q)
    min_len = int(0.7 * len_before)
    max_len = int(1.5 * len_before)

    full_prompt = (
        f"{_IMPROVE_SYSTEM_Q}\n\n"
        f"Modèle cible : {model_name}\n\n"
        f"CONTRAINTE DE TAILLE : Le prompt modifié doit contenir entre {min_len} et {max_len} "
        f"caractères (actuellement {len_before}). Ne supprime pas de sections entières.\n\n"
        f"Prompt actuel :\n{prompt_q}\n\n"
        f"Question ayant échoué aux métriques :\n"
        f"Question : {step1.question}\n"
        f"Réponse correcte : {step1.correct_answer}\n\n"
        f"Critères échoués :\n" + "\n".join(f"  - {f}" for f in failures) + "\n\n"
        f"→ Améliore le prompt pour que ce modèle génère de meilleures questions."
    )
    result = claude(full_prompt)
    raw = result.content.strip() if result.content else ""
    if not raw:
        print("    [Qwen] ⚠ réponse vide, prompt conservé")
        return prompt_q

    improved = _extract_improved_prompt(
        raw, prompt_q,
        required_sentinels=[_SENTINEL, _SECTION_Q_START, _SECTION_Q_END],
    )
    return improved if improved is not None else prompt_q


def improve_distractor_prompt(
    prompt_d: str,
    step1: MCQStep1,
    step2: MCQStep2,
    b3: dict,
    model_name: str,
) -> str:
    """Ask Qwen to improve the distractor-generation section of prompt_d."""
    distractors_text = [step2.distractor_1, step2.distractor_2, step2.distractor_3]
    score_lines = ""
    for i, (text, score, justif) in enumerate(
        zip(distractors_text, b3.get("scores", []), b3.get("justifs", [])), start=1
    ):
        score_lines += f"  - Distracteur {i} « {text} » → {score}/5 — {justif}\n"

    len_before = len(prompt_d)
    min_len = int(0.7 * len_before)
    max_len = int(1.5 * len_before)

    full_prompt = (
        f"{_IMPROVE_SYSTEM_D}\n\n"
        f"Modèle cible : {model_name}\n\n"
        f"CONTRAINTE DE TAILLE : Le prompt modifié doit contenir entre {min_len} et {max_len} "
        f"caractères (actuellement {len_before}). Ne supprime pas de sections entières.\n\n"
        f"Prompt actuel :\n{prompt_d}\n\n"
        f"QCM ayant échoué (distracteurs de mauvaise qualité) :\n"
        f"Question (gelée) : {step1.question}\n"
        f"Réponse correcte : {step1.correct_answer}\n"
        f"Scores distracteurs :\n{score_lines}"
        f"→ Améliore le prompt pour que ce modèle génère de meilleurs distracteurs."
    )
    result = claude(full_prompt)
    raw = result.content.strip() if result.content else ""
    if not raw:
        print("    [Qwen] ⚠ réponse vide, prompt conservé")
        return prompt_d

    improved = _extract_improved_prompt(
        raw, prompt_d,
        required_sentinels=[_SENTINEL, _SENTINEL_Q, _SENTINEL_ANS, _SECTION_D_START, _SECTION_D_END],
    )
    return improved if improved is not None else prompt_d


# ─────────────────────────────────────────────────────────────────
# Optimization loop (one model)
# ─────────────────────────────────────────────────────────────────

def optimize_model(
    save_name: str,
    model_id: str,
    sheets: list[dict],
    initial_prompt_q: str,
    initial_prompt_d: str,
    openai_client: OpenAI,
    b3_system_prompt: str,
    max_q_attempts: int,
    max_d_attempts: int,
    output_dir: Path,
    tracer: Tracer,
) -> tuple[str, str]:
    model_dir = output_dir / save_name
    model_dir.mkdir(parents=True, exist_ok=True)

    tracer.model_start(save_name, model_id, len(sheets), max_q_attempts + max_d_attempts)

    t_load = time.monotonic()
    pipe = _load_pipeline(model_id)
    tracer.model_loaded(save_name, time.monotonic() - t_load)

    prompt_q = initial_prompt_q
    prompt_d = initial_prompt_d
    log: list[dict] = []
    t_model = time.monotonic()

    try:
        for sheet_idx, sheet in enumerate(sheets):
            content: str = sheet["content_raw"]
            sheet_id = sheet.get("id", sheet_idx)
            tracer.sheet_start(save_name, sheet_idx, sheet_id, len(sheets))

            prompt_q_entree = prompt_q
            prompt_d_entree = prompt_d

            # ── Stage 1 : question quality ────────────────────────
            question_validated = False
            step1 = None
            q_attempt = 0
            for q_attempt in range(max_q_attempts):
                print(f"    [{_ts()}] Q tentative {q_attempt+1}/{max_q_attempts}", end="", flush=True)

                t_gen = time.monotonic()
                step1 = generate_question(content, prompt_q, pipe, save_name)
                ok_q = step1 is not None
                dur_gen = time.monotonic() - t_gen
                print(f" {'ok' if ok_q else 'INVALIDE'} ({dur_gen:.1f}s)", end="")
                if step1 is None:
                    print()
                    break

                qresult = evaluate_question_quality(step1.question, content)
                tracer.log(
                    "question_quality",
                    save_name=save_name, sheet_idx=sheet_idx, attempt=q_attempt,
                    **qresult,
                )
                badge = "✅" if qresult["passes"] else "❌"
                print(
                    f"  → Q={badge} A1={qresult['a1']} A2={qresult['a2']} "
                    f"A3={qresult['a3']:.2f} A4={qresult['a4']} B1={qresult['b1']:.2f}"
                )

                if qresult["passes"]:
                    question_validated = True
                    break

                if q_attempt < max_q_attempts - 1:
                    prev_q = prompt_q
                    t_improve = time.monotonic()
                    prompt_q = improve_question_prompt(prompt_q, step1, qresult, model_id)
                    tracer.claude_improve(
                        save_name, sheet_idx, q_attempt,
                        time.monotonic() - t_improve, prev_q, prompt_q,
                        stage="question",
                    )

            if not question_validated:
                prompt_q = prompt_q_entree
                tracer.log("prompt_rollback", save_name=save_name, sheet_idx=sheet_idx, stage="question")
                print(f"    [{_ts()}] prompt_q rollback → état avant fiche {sheet_idx}")
                tracer.sheet_done(save_name, sheet_idx, sheet_id, False, q_attempt + 1)
                log.append({
                    "sheet_idx": sheet_idx, "sheet_id": sheet_id,
                    "question_attempts": q_attempt + 1, "distractor_attempts": 0,
                    "passed": False,
                })
                continue

            # ── Stage 2 : distracteur quality (question gelée) ────
            distractor_passed = False
            d_attempt = 0
            best_step2: MCQStep2 | None = None
            best_scores: list[int] = [0, 0, 0]
            best_justifs: list[str] = ["", "", ""]
            for d_attempt in range(max_d_attempts):
                print(f"    [{_ts()}] D tentative {d_attempt+1}/{max_d_attempts}", end="", flush=True)

                t_gen = time.monotonic()
                step2 = generate_distractors(
                    content, step1.question, step1.correct_answer,
                    prompt_d, pipe, save_name,
                )
                ok_d = step2 is not None
                dur_gen = time.monotonic() - t_gen
                print(f" {'ok' if ok_d else 'INVALIDE'} ({dur_gen:.1f}s)", end="")
                if step2 is None:
                    print()
                    break

                mcq, d_slots = assemble_mcq(step1, step2)
                t_b3 = time.monotonic()
                b3 = evaluate_b3(mcq, content, openai_client, b3_system_prompt)
                dur_b3 = time.monotonic() - t_b3

                # Map GPT-4o scores (ordered by slot a→d, excl. correct) back to distractor_1/2/3
                if b3["scores"]:
                    non_correct = [s for s in ["a", "b", "c", "d"] if s != mcq.correct_option]
                    slot_to_score = dict(zip(non_correct, b3["scores"]))
                    slot_to_justif = dict(zip(non_correct, b3["justifs"]))
                    d_scores = [slot_to_score.get(d_slots[i], 0) for i in range(3)]
                    d_justifs = [slot_to_justif.get(d_slots[i], "") for i in range(3)]
                else:
                    d_scores = [0, 0, 0]
                    d_justifs = ["", "", ""]

                # Greedy per-distractor update: keep a distractor only if its score improves
                if best_step2 is None:
                    best_step2 = step2
                    best_scores = d_scores[:]
                    best_justifs = d_justifs[:]
                else:
                    for i in range(3):
                        if d_scores[i] > best_scores[i]:
                            setattr(best_step2, f"distractor_{i+1}",
                                    getattr(step2, f"distractor_{i+1}"))
                            setattr(best_step2, f"distractor_{i+1}_comment",
                                    getattr(step2, f"distractor_{i+1}_comment"))
                            best_scores[i] = d_scores[i]
                            best_justifs[i] = d_justifs[i]

                effective_passes = sum(s >= 4 for s in best_scores) >= 2
                tracer.b3_result(
                    save_name, sheet_idx, d_attempt,
                    dur_b3,
                    effective_passes, best_scores, best_justifs,
                )

                if effective_passes:
                    distractor_passed = True
                    break

                if d_attempt < max_d_attempts - 1:
                    prev_d = prompt_d
                    t_improve = time.monotonic()
                    best_b3 = {"passes": False, "scores": best_scores, "justifs": best_justifs}
                    prompt_d = improve_distractor_prompt(prompt_d, step1, best_step2, best_b3, model_id)
                    tracer.claude_improve(
                        save_name, sheet_idx, d_attempt,
                        time.monotonic() - t_improve, prev_d, prompt_d,
                        stage="distractor",
                    )

            if not distractor_passed:
                prompt_d = prompt_d_entree
                tracer.log("prompt_rollback", save_name=save_name, sheet_idx=sheet_idx, stage="distractor")
                print(f"    [{_ts()}] prompt_d rollback → état avant fiche {sheet_idx}")

            passed = distractor_passed
            tracer.sheet_done(save_name, sheet_idx, sheet_id, passed, d_attempt + 1)
            log.append({
                "sheet_idx": sheet_idx, "sheet_id": sheet_id,
                "question_attempts": q_attempt + 1,
                "distractor_attempts": d_attempt + 1,
                "passed": passed,
            })

    finally:
        _unload_pipeline(pipe)

    # Write intermediate debug files
    (model_dir / "final_prompt_question.txt").write_text(prompt_q, encoding="utf-8")
    (model_dir / "final_prompt_distractor.txt").write_text(prompt_d, encoding="utf-8")

    # Merge into single deployment prompt
    try:
        final_prompt = build_final_prompt(prompt_q, prompt_d)
    except ValueError as e:
        print(f"  [{_ts()}] ⚠ merge échoué ({e}) — fallback: prompt_q utilisé")
        final_prompt = prompt_q
    (model_dir / "final_prompt.txt").write_text(final_prompt, encoding="utf-8")

    (model_dir / "optimization_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    passed_count = sum(1 for e in log if e["passed"])
    tracer.model_done(save_name, time.monotonic() - t_model, passed_count, len(sheets))
    return prompt_q, prompt_d


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
        "--max-q-attempts", type=int, default=3, dest="max_q_attempts",
        help="Max attempts for question stage per sheet (default: 3)",
    )
    p.add_argument(
        "--max-d-attempts", type=int, default=5, dest="max_d_attempts",
        help="Max attempts for distractor stage per sheet (default: 5)",
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

    initial_prompt_q = _build_initial_question_prompt()
    initial_prompt_d = _build_initial_distractor_prompt()
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
                    initial_prompt_q=initial_prompt_q,
                    initial_prompt_d=initial_prompt_d,
                    openai_client=openai_client,
                    b3_system_prompt=b3_prompt,
                    max_q_attempts=args.max_q_attempts,
                    max_d_attempts=args.max_d_attempts,
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
