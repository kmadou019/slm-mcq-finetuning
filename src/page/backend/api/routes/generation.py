"""
Generation Routes - API endpoints for custom MCQ generation via Ollama GPU.
"""
import asyncio
import io
import json
import os
import uuid
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, ConfigDict
from pypdf import PdfReader
from sqlalchemy.orm import Session

from ..utils.dependencies import get_current_user
from ..utils.generate_mcq_GPU import generate_mcq, validate_mcq, MCQQuestion
from ..models.auth import User
from ..models.db_models import MCQAssignment as DBMCQAssignment
from database import get_db, SessionLocal
from eval.eval_dataframe import eval_dataframe

router = APIRouter()

BACKEND_DATA_DIR = Path(__file__).parent.parent.parent / "data"
CUSTOM_MCQS_DIR = BACKEND_DATA_DIR / "custom_mcqs"

# In-memory job store: {job_id: {status, user_id, mcq_count, error}}
_jobs: Dict[str, Dict[str, Any]] = {}


class GenerationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    content: str
    model_save_name: str
    prompt_template: str | None = None



def _build_mcqs_from_response(job_id: str, content: str, model_save_name: str, mcq: MCQQuestion) -> list:
    """Convert a validated MCQQuestion (Ollama response) into 1 MCQCard dict."""
    section_a_placeholder = [
        {"check_id": "A1", "description": "Is a question", "result": "PASS",
         "status": "not_checked", "confidence": None, "threshold": "True", "notes": "N/A"},
        {"check_id": "A2", "description": "No leading negation", "result": "PASS",
         "status": "not_checked", "confidence": None, "threshold": "False", "notes": "N/A"},
        {"check_id": "A3", "description": "Originality", "result": "PASS",
         "status": "not_checked", "confidence": None, "threshold": ">= 0.75", "notes": "N/A"},
        {"check_id": "A4", "description": "Readability (FK grade)", "result": "PASS",
         "status": "not_checked", "confidence": None, "threshold": ">= 12", "notes": "N/A"},
    ]
    section_b_placeholder = [
        {"check_id": "B1", "description": "Disclosure (answer leakage)", "result": "N/A",
         "status": "not_checked", "confidence": None, "score": "True/False", "notes": "N/A"},
        {"check_id": "B2", "description": "Relevance to material", "result": "N/A",
         "status": "not_checked", "confidence": None, "score": "0-1", "notes": "N/A"},
        {"check_id": "B3", "description": "Distractor plausibility", "result": "N/A",
         "status": "not_checked", "confidence": None, "score": "True/False", "notes": "N/A"},
        {"check_id": "B4", "description": "Difficulty appropriateness", "result": "N/A",
         "status": "not_checked", "confidence": None, "score": "low/med/high", "notes": "N/A"},
    ]
    return [{
        "item_id": f"{job_id}-1",
        "source_material": job_id,
        "generator_info": model_save_name,
        "output_format": "JSON",
        "mcq_question": mcq.question,
        "question_comment": mcq.question_comment,
        "options": {"A": mcq.option_a, "B": mcq.option_b, "C": mcq.option_c, "D": mcq.option_d},
        "option_comments": {"A": mcq.option_a_comment, "B": mcq.option_b_comment, "C": mcq.option_c_comment, "D": mcq.option_d_comment},
        "correct_option": mcq.correct_option.upper(),
        "section_a_checks": section_a_placeholder,
        "section_b_checks": section_b_placeholder,
        "decision_policy": "Accept if all hard constraints pass and no critical AI-judge dimension fails.",
        "final_decision": "ACCEPT",
        "audit_trail": f"Généré depuis contenu personnalisé. Modèle : {model_save_name}. Job : {job_id}.",
        "lisa_texte_brut": content,
        "lisa_metadata": {
            "identifiant": job_id,
            "rang": "B",
            "rubrique": "Contenu personnalisé",
            "intitule": "Question générée depuis votre contenu",
            "item_parent": "Contenu évaluateur",
            "description": "MCQ généré automatiquement depuis le matériel pédagogique fourni par l'évaluateur.",
            "contenu": content,
        },
    }]


def _evaluate_mcqs(mcq_cards: list, content: str) -> list:
    """
    Run the quality eval pipeline on the 2 MCQ cards and update their checks.
    Converts MCQCards to a DataFrame, calls eval_dataframe, maps results back.
    """
    # Re-read .env each time so the key is picked up even if added after startup
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    openai_key = os.environ.get("OPENAI_API_KEY") or ""
    print(f"[eval] OpenAI key present: {bool(openai_key)}")

    # Load system prompts
    prompts_path = Path(__file__).parent.parent.parent / "eval" / "prompts.json"
    with open(prompts_path, "r", encoding="utf-8") as f:
        system_prompts = json.load(f)

    # Build a flat DataFrame from MCQCards (eval_dataframe expects CSV-style columns)
    rows = []
    for card in mcq_cards:
        rows.append({
            "question":       card["mcq_question"],
            "option_a":       card["options"]["A"],
            "option_b":       card["options"]["B"],
            "option_c":       card["options"]["C"],
            "option_d":       card["options"]["D"],
            "correct_option": card["correct_option"].lower(),  # eval expects lowercase
            "content_raw":    content,
        })
    df = pd.DataFrame(rows)

    has_openai = bool(openai_key)

    df = eval_dataframe(
        df_merged=df,
        openai_key=openai_key,
        # LLM-based checks only if we have an OpenAI key
        compute_answerability       =has_openai,
        answerability_system_prompt =system_prompts.get("answerability_prompt") if has_openai else None,
        compute_disclosure          =has_openai,
        disclosure_system_prompt    =system_prompts.get("disclosure_prompt") if has_openai else None,
        compute_difficulty          =has_openai,
        difficulty_system_prompt    =system_prompts.get("difficulty_prompt") if has_openai else None,
        compute_distractors_quality =has_openai,
        distractors_quality_system_prompt=system_prompts.get("distractors_quality_prompt") if has_openai else None,
        # Non-LLM checks always run
        compute_originality=True,
        compute_readability=True,
        compute_negation=True,
        compute_is_question=True,
        compute_relevance=True,
        compute_ambiguity=True,
        # Column names
        question_col="question",
        option_a_col="option_a",
        option_b_col="option_b",
        option_c_col="option_c",
        option_d_col="option_d",
        correct_option_col="correct_option",
        lisa_sheet_col="content_raw",
        output_file_path=None,
    )

    # Map eval results back to section_a_checks / section_b_checks in each card
    for i, card in enumerate(mcq_cards):
        row = df.iloc[i]

        def _get(col, default=None):
            v = row.get(col, default) if hasattr(row, "get") else row[col] if col in df.columns else default
            return v if pd.notna(v) else default

        is_q       = bool(_get("is_question", True))
        negation   = bool(_get("starts_with_negation", False))
        orig       = _get("originality", None)
        readab     = _get("readability", None)
        disclosure = _get("disclosure", None)
        relevance  = _get("relevance", None)
        distractors= _get("distractors_quality", None)
        difficulty = _get("difficulty", None)

        def _result(condition, checked=True):
            if not checked:
                return "N/A"
            return "PASS" if condition else "FAIL"

        card["section_a_checks"] = [
            {
                "check_id": "A1", "description": "Is a question",
                "result": _result(is_q),
                "status": "not_checked", "confidence": None, "threshold": "True",
                "notes": f"{'Yes' if is_q else 'No'}",
            },
            {
                "check_id": "A2", "description": "No leading negation",
                "result": _result(not negation),
                "status": "not_checked", "confidence": None, "threshold": "False",
                "notes": f"Negation: {negation}",
            },
            {
                "check_id": "A3", "description": "Originality",
                "result": _result(orig is not None and orig >= 0.75) if orig is not None else "N/A",
                "status": "not_checked",
                "confidence": None, "threshold": ">= 0.75",
                "notes": f"Score: {round(orig, 3) if orig is not None else 'N/A'}",
            },
            {
                "check_id": "A4", "description": "Readability (FK grade)",
                "result": _result(readab is not None and readab >= 12) if readab is not None else "N/A",
                "status": "not_checked",
                "confidence": None, "threshold": ">= 12",
                "notes": f"Score: {round(readab, 1) if readab is not None else 'N/A'}",
            },
        ]

        card["section_b_checks"] = [
            {
                "check_id": "B1", "description": "Disclosure (answer leakage)",
                "result": _result(disclosure in (None, "False", False, "false")) if has_openai else "N/A",
                "status": "not_checked",
                "confidence": None, "score": "True/False",
                "notes": f"Score: {disclosure}" if has_openai else "N/A (no OpenAI key)",
            },
            {
                "check_id": "B2", "description": "Relevance to material",
                "result": _result(relevance is not None and relevance >= 0.5) if relevance is not None else "N/A",
                "status": "not_checked",
                "confidence": None, "score": "0-1",
                "notes": f"Score: {round(relevance, 3) if relevance is not None else 'N/A'}",
            },
            {
                "check_id": "B3", "description": "Distractor plausibility",
                "result": _result(distractors) if has_openai else "N/A",
                "status": "not_checked",
                "confidence": None, "score": "True/False",
                "notes": f"Score: {distractors}" if has_openai else "N/A (no OpenAI key)",
            },
            {
                "check_id": "B4", "description": "Difficulty appropriateness",
                "result": "PASS" if difficulty else ("FAIL" if difficulty is not None else "N/A"),
                "status": "not_checked",
                "confidence": None, "score": "low/med/high",
                "notes": f"Judge: {difficulty}" if has_openai else "N/A (no OpenAI key)",
            },
        ]

        # Recompute final_decision: ACCEPT unless a section A check is FAIL
        a_fails = [c for c in card["section_a_checks"] if c["result"] == "FAIL"]
        card["final_decision"] = "REJECT" if a_fails else "ACCEPT"

    return mcq_cards


async def _run_generation(job_id: str, user_id: int, content: str, model_save_name: str, prompt_template: str | None = None):
    """Background task: call Ollama on remote GPU, evaluate MCQs, then create assignments."""
    try:
        full_prompt = (prompt_template or "").replace("{content}", content)
        raw_response = await asyncio.to_thread(generate_mcq, full_prompt, model_name=model_save_name)
        mcq = validate_mcq(raw_response)
        if mcq is None:
            raise ValueError(f"Ollama response failed validation: {raw_response}")

        real_mcqs = _build_mcqs_from_response(job_id, content, model_save_name, mcq)

        print(f"[generation] Running eval pipeline on {len(real_mcqs)} MCQs...")
        real_mcqs = await asyncio.to_thread(_evaluate_mcqs, real_mcqs, content)
        print(f"[generation] Eval done.")

        # Persist MCQ cards to JSON
        CUSTOM_MCQS_DIR.mkdir(parents=True, exist_ok=True)
        with open(CUSTOM_MCQS_DIR / f"{job_id}.json", "w", encoding="utf-8") as f:
            json.dump(real_mcqs, f, ensure_ascii=False, indent=2)

        # Create DB assignments
        db = SessionLocal()
        try:
            for i in range(len(real_mcqs)):
                mcq_id = f"{job_id}-{i + 1}"
                exists = db.query(DBMCQAssignment).filter(
                    DBMCQAssignment.user_id == user_id,
                    DBMCQAssignment.mcq_id == mcq_id,
                    DBMCQAssignment.model == "custom",
                ).first()
                if not exists:
                    db.add(DBMCQAssignment(
                        user_id=user_id,
                        mcq_id=mcq_id,
                        model="custom",
                        status="pending",
                    ))
            db.commit()
        finally:
            db.close()

        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["mcq_count"] = len(real_mcqs)
        print(f"✅ Generation done — job {job_id}: {len(real_mcqs)} MCQs assigned to user {user_id}")

    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(e)
        print(f"❌ Generation failed — job {job_id}: {e}")


# ============================================================================
# ROUTES
# ============================================================================

@router.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    """Extrait le texte brut d'un fichier PDF uploadé."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Le fichier doit être un PDF.")
    content = await file.read()
    reader = PdfReader(io.BytesIO(content))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Impossible d'extraire du texte de ce PDF (peut-être scanné).")
    return {"text": text}


@router.post("/generate")
async def start_generation(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Start MCQ generation from custom educational content.
    Returns a job_id immediately; poll GET /generate/{job_id} for status.
    """
    if len(request.content.strip()) < 100:
        raise HTTPException(status_code=400, detail="Le contenu doit contenir au moins 100 caractères.")

    job_id = f"CSTM-{uuid.uuid4().hex[:7].upper()}"
    _jobs[job_id] = {
        "status": "running",
        "user_id": current_user.id,
        "mcq_count": 0,
        "error": None,
        "prompt_template": request.prompt_template,  # stored for real GPU use
    }

    background_tasks.add_task(
        _run_generation,
        job_id=job_id,
        user_id=current_user.id,
        content=request.content,
        model_save_name=request.model_save_name,
        prompt_template=request.prompt_template,
    )

    print(f"🚀 Generation job started — {job_id} for user {current_user.username} (model: {request.model_save_name})")
    return {"job_id": job_id}


@router.get("/generate/{job_id}")
async def get_generation_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Poll the status of a generation job.
    Response: { job_id, status: 'running'|'done'|'error', mcq_count, error }
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable.")
    if job["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Accès refusé.")

    return {
        "job_id": job_id,
        "status": job["status"],
        "mcq_count": job["mcq_count"],
        "error": job.get("error"),
    }
