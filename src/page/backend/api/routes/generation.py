"""
Generation Routes - API endpoints for custom MCQ generation (mock for demo)
"""
import asyncio
import json
import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from ..utils.dependencies import get_current_user
from ..models.auth import User
from ..models.db_models import MCQAssignment as DBMCQAssignment
from database import get_db, SessionLocal

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


def _build_mock_mcqs(job_id: str, content: str, model_save_name: str) -> list:
    """Build 2 hardcoded realistic mock MCQ cards for demo purposes."""
    section_a_pass = [
        {"check_id": "A1", "description": "Question mark present", "result": "PASS",
         "status": "not_checked", "confidence": None, "threshold": "required", "notes": "Question format validated"},
        {"check_id": "A2", "description": "No leading negation", "result": "PASS",
         "status": "not_checked", "confidence": None, "threshold": "required", "notes": "No negation"},
        {"check_id": "A3", "description": "Originality (integral)", "result": "PASS",
         "status": "not_checked", "confidence": None, "threshold": "≥ 0.75", "notes": "Score: 0.87"},
        {"check_id": "A4", "description": "Readability (FK grade)", "result": "PASS",
         "status": "not_checked", "confidence": None, "threshold": "≥ 12", "notes": "Score: 14.2"},
    ]

    section_b_pass = [
        {"check_id": "B1", "description": "Disclosure (answer leakage)", "result": "PASS",
         "status": "not_checked", "confidence": None, "score": "True/False", "notes": "Score: False"},
        {"check_id": "B2", "description": "Relevance to material", "result": "PASS",
         "status": "not_checked", "confidence": None, "score": "0-1", "notes": "Score: 0.91"},
        {"check_id": "B3", "description": "Distractor plausibility", "result": "PASS",
         "status": "not_checked", "confidence": None, "score": "True/False", "notes": "Score: True"},
        {"check_id": "B4", "description": "Difficulty appropriateness", "result": "PASS",
         "status": "not_checked", "confidence": None, "score": "low/med/high", "notes": "Judge: medium"},
    ]

    section_b_warn = [
        {"check_id": "B1", "description": "Disclosure (answer leakage)", "result": "WARN",
         "status": "not_checked", "confidence": None, "score": "True/False", "notes": "Score: True"},
        {"check_id": "B2", "description": "Relevance to material", "result": "PASS",
         "status": "not_checked", "confidence": None, "score": "0-1", "notes": "Score: 0.78"},
        {"check_id": "B3", "description": "Distractor plausibility", "result": "PASS",
         "status": "not_checked", "confidence": None, "score": "True/False", "notes": "Score: True"},
        {"check_id": "B4", "description": "Difficulty appropriateness", "result": "PASS",
         "status": "not_checked", "confidence": None, "score": "low/med/high", "notes": "Judge: medium"},
    ]

    common_metadata = {
        "identifiant": job_id,
        "rang": "B",
        "rubrique": "Contenu personnalisé",
        "intitule": "Question générée depuis votre contenu",
        "item_parent": "Contenu évaluateur",
        "description": "MCQ généré automatiquement depuis le matériel pédagogique fourni par l'évaluateur.",
        "contenu": content,
    }

    mcq1 = {
        "item_id": f"{job_id}-1",
        "source_material": job_id,
        "generator_info": model_save_name,
        "output_format": "JSON",
        "mcq_question": "Parmi les propositions suivantes concernant la physiopathologie décrite dans le contenu fourni, laquelle est correcte ?",
        "question_comment": "Cette question évalue la compréhension des mécanismes fondamentaux présentés dans le cours.",
        "options": {
            "A": "L'activation du système sympathique entraîne une vasoconstriction périphérique.",
            "B": "La libération d'acétylcholine provoque une tachycardie réflexe.",
            "C": "L'hypoxie tissulaire stimule directement la libération de catécholamines médullosurrénaliennes.",
            "D": "L'aldostérone agit principalement sur la réabsorption du calcium au niveau du tubule proximal.",
        },
        "option_comments": {
            "A": "Correct : la noradrénaline libérée par les terminaisons sympathiques active les récepteurs α1 vasculaires, entraînant une vasoconstriction.",
            "B": "Incorrect : l'acétylcholine a un effet chronotrope négatif (bradycardie) via les récepteurs M2 cardiaques.",
            "C": "Incorrect : l'hypoxie stimule leglomus carotidien (chémorécepteurs périphériques), ce qui active indirectement le système sympathique.",
            "D": "Incorrect : l'aldostérone agit principalement sur la réabsorption du sodium (et l'excrétion du potassium) au niveau du tube collecteur.",
        },
        "correct_option": "A",
        "section_a_checks": section_a_pass,
        "section_b_checks": section_b_pass,
        "decision_policy": "Accept if all hard constraints pass and no critical AI-judge dimension fails.",
        "final_decision": "ACCEPT",
        "audit_trail": f"Généré depuis contenu personnalisé. Modèle : {model_save_name}. Job : {job_id}.",
        "lisa_texte_brut": content,
        "lisa_metadata": common_metadata,
    }

    mcq2 = {
        "item_id": f"{job_id}-2",
        "source_material": job_id,
        "generator_info": model_save_name,
        "output_format": "JSON",
        "mcq_question": "Dans le contexte du contenu fourni, quelle affirmation est la plus exacte concernant la prise en charge thérapeutique ?",
        "question_comment": "Question portant sur l'application clinique des notions exposées dans le matériel pédagogique.",
        "options": {
            "A": "Les bêtabloquants sont contre-indiqués en phase de décompensation cardiaque aiguë.",
            "B": "Les inhibiteurs de l'ECA sont préférés aux sartans en cas de toux chronique sous traitement.",
            "C": "La digoxine est le traitement de première intention dans la fibrillation atriale paroxystique.",
            "D": "Les diurétiques de l'anse augmentent la réabsorption du sodium au niveau du tubule proximal.",
        },
        "option_comments": {
            "A": "Correct : en phase aiguë de décompensation, les bêtabloquants peuvent aggraver l'état hémodynamique et sont initialement contre-indiqués.",
            "B": "Incorrect : la toux est un effet indésirable des IEC, qui justifie précisément le passage aux sartans (ARA II), non l'inverse.",
            "C": "Incorrect : les anticoagulants et la cardioversion sont prioritaires ; la digoxine est réservée à des cas particuliers (FA + insuffisance cardiaque).",
            "D": "Incorrect : les diurétiques de l'anse (furosémide) bloquent le cotransporteur NKCC2 au niveau de l'anse de Henlé, non le tubule proximal.",
        },
        "correct_option": "A",
        "section_a_checks": section_a_pass,
        "section_b_checks": section_b_warn,
        "decision_policy": "Accept if all hard constraints pass and no critical AI-judge dimension fails.",
        "final_decision": "REVISE",
        "audit_trail": f"Généré depuis contenu personnalisé. Modèle : {model_save_name}. Job : {job_id}.",
        "lisa_texte_brut": content,
        "lisa_metadata": common_metadata,
    }

    return [mcq1, mcq2]


async def _run_mock_generation(job_id: str, user_id: int, content: str, model_save_name: str, prompt_template: str | None = None):
    """Background task: simulate GPU generation delay then create mock MCQ assignments."""
    # Simulate processing time (~3s instead of several minutes)
    await asyncio.sleep(3)

    # Construire le prompt final : insérer le contenu dans le template
    # TODO: envoyer full_prompt au serveur GPU (remplacer le mock ci-dessous)
    full_prompt = (prompt_template or "").replace("{content}", content)

    try:
        mock_mcqs = _build_mock_mcqs(job_id, content, model_save_name)

        # Persist mock MCQ cards to JSON
        CUSTOM_MCQS_DIR.mkdir(parents=True, exist_ok=True)
        with open(CUSTOM_MCQS_DIR / f"{job_id}.json", "w", encoding="utf-8") as f:
            json.dump(mock_mcqs, f, ensure_ascii=False, indent=2)

        # Create DB assignments for the 2 mock MCQs
        db = SessionLocal()
        try:
            for i in range(len(mock_mcqs)):
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
        _jobs[job_id]["mcq_count"] = len(mock_mcqs)
        print(f"✅ Mock generation done — job {job_id}: {len(mock_mcqs)} MCQs assigned to user {user_id}")

    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(e)
        print(f"❌ Mock generation failed — job {job_id}: {e}")


# ============================================================================
# ROUTES
# ============================================================================

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
    (Mock implementation — replaces real GPU endpoint for demo.)
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
        _run_mock_generation,
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
