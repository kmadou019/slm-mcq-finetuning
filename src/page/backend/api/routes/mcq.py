"""
MCQ Routes - API endpoints for MCQ evaluation
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import List, Dict, Any, Optional
import pandas as pd
import re
import json
import random
from pathlib import Path
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..utils.dependencies import get_current_user
from ..models.auth import User
from ..models.db_models import MCQAssignment as DBMCQAssignment
from database import get_db

router = APIRouter()

# Chemins vers les fichiers
BACKEND_DATA_DIR = Path(__file__).parent.parent.parent / "data"  # Données app web (backend/)
CSV_DIR = BACKEND_DATA_DIR / "mcqs"  # Fichiers CSV des MCQ
CSV_PATH = CSV_DIR / "qwen3_8b_pdapt_slerp.csv"  # CSV par défaut
ASSIGNMENTS_PATH = BACKEND_DATA_DIR / "assignments.json"  # Assignations par utilisateur
GLOBAL_TRACKER_PATH = BACKEND_DATA_DIR / "global_assignment_tracker.json"  # Tracker global par modèle
LISA_SHEETS_PATH = BACKEND_DATA_DIR / "lisa_sheets.csv"  # Lisa sheets de référence

# Cache pour les lisa sheets (chargé une seule fois)
_lisa_sheets_cache: Dict[str, str] | None = None

def get_lisa_content(mcq_id: str) -> str:
    """Lookup content_raw from lisa_sheets.csv by MCQ id (with trailing - stripped)"""
    global _lisa_sheets_cache
    if _lisa_sheets_cache is None:
        _lisa_sheets_cache = {}
        if LISA_SHEETS_PATH.exists():
            try:
                df = pd.read_csv(LISA_SHEETS_PATH, engine='python', quotechar='"', on_bad_lines='skip')
                for _, row in df.iterrows():
                    _lisa_sheets_cache[str(row.get('id', ''))] = str(row.get('content_raw', ''))
                print(f"Loaded {len(_lisa_sheets_cache)} lisa sheets from {LISA_SHEETS_PATH}")
            except Exception as e:
                print(f"Error loading lisa sheets: {e}")
    # Try exact match, then stripped match
    result = _lisa_sheets_cache.get(mcq_id, '')
    if not result:
        result = _lisa_sheets_cache.get(mcq_id.rstrip('-'), '')
    return result

# Modèles disponibles
AVAILABLE_MODELS = [
    "llama3_1_8b", "openbiollm_8b", "gemma2_9b",
    "medGemma_4b", "medGemma_27b", "qwen3_8b",
    "mistral_7b", "eurollm_9b", "apertus_8B",
    "qwen3_0.6b", "qwen3_1_7b", "qwen3_4b",
    "qwen3_8b_pdapt_slerp", "qwen3_4b_pdapt_slerp",
    "qwen3_1_7b_pdapt_slerp", "qwen3_0.6b_pdapt_slerp"
]


# ============================================================================
# MODELS PYDANTIC
# ============================================================================

class SectionCheck(BaseModel):
    check_id: str
    description: str
    result: str  # 'PASS' ou 'WARN'
    status: str  # 'not_checked', 'validated', 'rejected'
    confidence: str | None
    threshold: str | None = None  # Pour Section A
    score: str | None = None      # Pour Section B
    notes: str | None = None


class LISAMetadata(BaseModel):
    identifiant: str
    rang: str
    rubrique: str
    intitule: str
    item_parent: str
    description: str
    contenu: str


class MCQCard(BaseModel):
    item_id: str
    source_material: str
    generator_info: str
    output_format: str
    mcq_question: str
    question_comment: str = ""
    options: Dict[str, str]
    option_comments: Dict[str, str] = {}
    correct_option: str
    section_a_checks: List[SectionCheck]
    section_b_checks: List[SectionCheck]
    decision_policy: str
    final_decision: str
    audit_trail: str
    lisa_texte_brut: str
    lisa_metadata: LISAMetadata | None = None


class ValidationRequest(BaseModel):
    section_a_checks: List[Dict[str, Any]]
    section_b_checks: List[Dict[str, Any]]
    human_decision: str  # 'ACCEPT' ou 'REJECT'
    human_feedback: str


class AssignmentRequest(BaseModel):
    username: str
    count: int  # Nombre de MCQs à assigner
    model: str = "llama3_1_8b"  # Modèle par défaut


# ============================================================================
# FONCTIONS UTILITAIRES (réutilisation de card.py)
# ============================================================================

def parser_lisa_sheet(lisa_texte_brut: str) -> Dict[str, str]:
    """
    Parse le texte brut d'une LISA Sheet et extrait les informations
    """
    data = {
        'identifiant': '',
        'rang': '',
        'intitule': '',
        'description': '',
        'rubrique': '',
        'item_parent': '',
        'contenu': ''
    }

    # Extraire l'identifiant
    match = re.search(r'\|Identifiant=([^\n|]+)', lisa_texte_brut)
    if match:
        data['identifiant'] = match.group(1).strip()

    # Extraire Item_parent
    match = re.search(r'\|Item_parent=([^\n|]*?)(?:\|Item_parent_short|$)', lisa_texte_brut, re.DOTALL)
    if match:
        data['item_parent'] = match.group(1).strip()

    # Extraire le rang
    match = re.search(r'\|Rang=([^\n|]+)', lisa_texte_brut)
    if match:
        data['rang'] = match.group(1).strip()

    # Extraire l'intitulé
    match = re.search(r'\|Intitulé=([^\n|]+)', lisa_texte_brut)
    if match:
        data['intitule'] = match.group(1).strip()

    # Extraire la description
    match = re.search(r'\|Description=([^\n|]+)', lisa_texte_brut)
    if match:
        data['description'] = match.group(1).strip()

    # Extraire la rubrique
    match = re.search(r'\|Rubrique=([^\n|]+)', lisa_texte_brut)
    if match:
        data['rubrique'] = match.group(1).strip()

    # Extraire le contenu texte après }}
    match = re.search(r'\}\}(.*)', lisa_texte_brut, re.DOTALL)
    if match:
        data['contenu'] = match.group(1).strip()

    return data


def evaluation_to_pass_warn(score, threshold) -> str:
    """
    Convertit un score numérique ou booléen en PASS ou WARN
    """
    if pd.isna(score):
        return "WARN"

    try:
        val = float(score)
        return "PASS" if val >= threshold else "WARN"
    except:
        if isinstance(score, str):
            return "PASS" if score.lower() in ['true', 'yes', 'pass', '1', "medium", 'low'] else "WARN"
        return "PASS" if score else "WARN"


def load_assignments() -> Dict[str, Any]:
    """
    Charger les assignations depuis le fichier JSON
    Format: {
        "username": [
            { "model": "qwen3_4b_pdapt_slerp", "mcq_ids": [...], "assigned_at": "..." },
            { "model": "qwen3_8b_pdapt_slerp", "mcq_ids": [...], "assigned_at": "..." }
        ]
    }
    """
    if not ASSIGNMENTS_PATH.exists():
        return {}

    try:
        with open(ASSIGNMENTS_PATH, 'r') as f:
            return json.load(f)
    except:
        return {}


def save_assignments(assignments: Dict[str, Dict[str, Any]]) -> None:
    """
    Sauvegarder les assignations dans le fichier JSON
    """
    ASSIGNMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ASSIGNMENTS_PATH, 'w') as f:
        json.dump(assignments, f, indent=2)


def load_global_tracker() -> Dict[str, Dict[str, Any]]:
    """
    Charger le tracker global des assignations par modèle
    Format: {
        "model_name": {
            "last_assigned_index": 39,  # Dernier index assigné (0-based)
            "total_available": 360,
            "assigned_count": 40
        }
    }
    """
    if not GLOBAL_TRACKER_PATH.exists():
        return {}

    try:
        with open(GLOBAL_TRACKER_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading global tracker: {e}")
        return {}


def save_global_tracker(tracker: Dict[str, Dict[str, Any]]) -> None:
    """
    Sauvegarder le tracker global dans le fichier JSON
    """
    GLOBAL_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_TRACKER_PATH, 'w') as f:
        json.dump(tracker, f, indent=2)


def get_csv_path_for_model(model: str) -> Path:
    """
    Obtenir le chemin du fichier CSV pour un modèle donné
    """
    csv_path = CSV_DIR / f"{model}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found for model '{model}' at {csv_path}")
    return csv_path


def assign_mcqs_to_user(username: str, count: int, model: str = "qwen3_4b_pdapt_slerp") -> Dict[str, Any]:
    """
    Assigner un nombre spécifique de MCQs à un utilisateur depuis un modèle spécifique
    Politique d'assignation: Les N prochains MCQs non assignés du fichier CSV
    (assignation séquentielle, pas aléatoire)

    Args:
        username: Nom d'utilisateur
        count: Nombre de MCQs à assigner
        model: Nom du modèle

    Returns:
        Dict avec model et mcq_ids
    """
    # Obtenir le chemin du CSV
    csv_path = get_csv_path_for_model(model)

    # Charger le CSV pour obtenir le nombre total
    df = pd.read_csv(csv_path, engine='python', quotechar='"', on_bad_lines='skip')
    total_mcqs = len(df)

    # Charger le tracker global
    global_tracker = load_global_tracker()

    # Initialiser ou mettre à jour le tracker pour ce modèle
    if model not in global_tracker:
        global_tracker[model] = {
            "last_assigned_index": -1,  # -1 signifie qu'aucun MCQ n'a été assigné
            "total_available": total_mcqs,
            "assigned_count": 0
        }
    else:
        # Toujours rafraîchir total_available depuis le CSV
        global_tracker[model]["total_available"] = total_mcqs

    # Récupérer le dernier index assigné
    last_index = global_tracker[model]["last_assigned_index"]
    start_index = last_index + 1

    # Vérifier qu'il reste assez de MCQs
    remaining = total_mcqs - start_index
    if remaining <= 0:
        raise ValueError(f"Plus de MCQs disponibles pour le modèle '{model}'. Total: {total_mcqs}, Déjà assignés: {start_index}")

    # Limiter le count au nombre restant
    actual_count = min(count, remaining)

    # Générer les indices séquentiels
    indices = list(range(start_index, start_index + actual_count))

    # Créer les IDs (index + 1 pour commencer à 1)
    mcq_ids = [f"MCQ-{i+1:06d}" for i in indices]

    # Mettre à jour le tracker global
    global_tracker[model]["last_assigned_index"] = start_index + actual_count - 1
    global_tracker[model]["assigned_count"] += actual_count
    save_global_tracker(global_tracker)

    # Charger les assignations existantes (format multi-modele)
    assignments = load_assignments()

    if username not in assignments:
        assignments[username] = []

    # Chercher un batch existant pour ce modele
    existing_batch = None
    for batch in assignments[username]:
        if batch.get("model") == model:
            existing_batch = batch
            break

    if existing_batch:
        existing_batch["mcq_ids"] = existing_batch.get("mcq_ids", []) + mcq_ids
        existing_batch["assigned_at"] = pd.Timestamp.now().isoformat()
    else:
        assignments[username].append({
            "model": model,
            "mcq_ids": mcq_ids,
            "assigned_at": pd.Timestamp.now().isoformat()
        })

    # Sauvegarder
    save_assignments(assignments)

    print(f"✅ Assigned {actual_count} MCQs from model '{model}' to user '{username}'")
    print(f"   Indices: {start_index} to {start_index + actual_count - 1}")
    print(f"   Remaining: {total_mcqs - (start_index + actual_count)}")

    return {
        "model": model,
        "mcq_ids": mcq_ids
    }


def build_mcq_card_from_row(row: pd.Series, index: int, model: str) -> Dict[str, Any]:
    """
    Construit une carte MCQ à partir d'une ligne du DataFrame
    """
    # Section A Checks
    section_a_checks = [
        {
            "check_id": "A1",
            "description": "Question mark present",
            "result": evaluation_to_pass_warn(row.get('is_question'), True),
            "status": "not_checked",
            "confidence": None,
            "threshold": "required",
            "notes": "Question format validated" if evaluation_to_pass_warn(row.get('is_question'), True) == "PASS" else "Not a question"
        },
        {
            "check_id": "A2",
            "description": "No leading negation",
            "result": "WARN" if row.get('starts_with_negation') == True or str(row.get('starts_with_negation')).lower() == 'true' else "PASS",
            "status": "not_checked",
            "confidence": None,
            "threshold": "required",
            "notes": "Negation detected" if row.get('starts_with_negation') == True or str(row.get('starts_with_negation')).lower() == 'true' else "No negation"
        },
        {
            "check_id": "A3",
            "description": "Originality (integral)",
            "result": evaluation_to_pass_warn(row.get('originality'), 0.75),
            "status": "not_checked",
            "confidence": None,
            "threshold": "≥ 0.75",
            "notes": f"Score: {row.get('originality', 'N/A')}"
        },
        {
            "check_id": "A4",
            "description": "Readability (FK grade)",
            "result": evaluation_to_pass_warn(row.get('readability'), 12),
            "status": "not_checked",
            "confidence": None,
            "threshold": "≥ 12",
            "notes": f"Score: {row.get('readability', 'N/A')}"
        }
    ]

    # Section B Checks
    difficulty = int(row.get('difficulty', 3))
    difficulty_mapping = {
        1: "low",
        2: "low",
        3: "medium",
        4: "high",
        5: "high"
    }

    # B3 — enrich with per-distractor scores if detail available
    distractor_detail_raw = row.get('distractors_quality_detail') or row.get('distractor_quality_detail')
    try:
        distractor_detail = json.loads(distractor_detail_raw) if distractor_detail_raw else None
    except Exception:
        distractor_detail = None

    if distractor_detail:
        scores  = distractor_detail.get("scores", [])
        justifs = distractor_detail.get("justifs", [])
        avg     = distractor_detail.get("avg", None)
        rang    = distractor_detail.get("rang", "B")
        b3_notes = f"Rang {rang} | avg={avg:.2f} | scores={scores}"
    else:
        b3_notes = f"Pass: {row.get('distractors_quality', 'N/A')}"

    # B5 — answerability: GPT-4o answered correctly from LISA context
    gpt_answer      = str(row.get('gpt_answer', '') or '').strip().lower()
    correct_opt     = str(row.get('correct_option', '') or '').strip().lower()
    answerability_ok = (gpt_answer == correct_opt) if gpt_answer else None

    # B6 — ambiguity: semantic similarity between correct answer and distractors
    ambiguity_val = row.get('ambiguity')
    try:
        ambiguity_float = float(ambiguity_val) if ambiguity_val is not None else None
    except (TypeError, ValueError):
        ambiguity_float = None

    section_b_checks = [
        {
            "check_id": "B1",
            "description": "Disclosure (answer leakage)",
            "result": evaluation_to_pass_warn(row.get('disclosure'), False),
            "status": "not_checked",
            "confidence": None,
            "score": "True/False",
            "notes": f"Score: {row.get('disclosure', 'N/A')}"
        },
        {
            "check_id": "B2",
            "description": "Relevance to material",
            "result": evaluation_to_pass_warn(row.get('relevance'), 0.8),
            "status": "not_checked",
            "confidence": None,
            "score": "0-1",
            "notes": f"Score: {row.get('relevance', 'N/A')}"
        },
        {
            "check_id": "B3",
            "description": "Distractor plausibility",
            "result": evaluation_to_pass_warn(row.get('distractors_quality', row.get('distractor_quality')), True),
            "status": "not_checked",
            "confidence": None,
            "score": "majority ≥ threshold & min ≥ 2",
            "notes": b3_notes,
        },
        {
            "check_id": "B4",
            "description": "Difficulty appropriateness",
            "result": evaluation_to_pass_warn(difficulty, "medium"),
            "status": "not_checked",
            "confidence": None,
            "score": "low/med/high",
            "notes": f"Judge: {difficulty_mapping.get(difficulty, 'medium')}"
        },
        {
            "check_id": "B5",
            "description": "Answerability (expert + context)",
            "result": ("PASS" if answerability_ok else ("FAIL" if answerability_ok is False else "N/A")),
            "status": "not_checked",
            "confidence": None,
            "score": "correct/incorrect",
            "notes": f"GPT-4o answered '{gpt_answer.upper()}', correct is '{correct_opt.upper()}'" if gpt_answer else "N/A",
        },
        {
            "check_id": "B6",
            "description": "Ambiguity",
            "result": ("PASS" if ambiguity_float is not None and ambiguity_float >= 0.3 else
                       ("FAIL" if ambiguity_float is not None else "N/A")),
            "status": "not_checked",
            "confidence": None,
            "score": "0-1 (≥ 0.3 = plausible)",
            "notes": f"Score: {round(ambiguity_float, 3) if ambiguity_float is not None else 'N/A'}",
        },
    ]

    # Parser LISA Sheet - utiliser content_raw du CSV, sinon fallback sur lisa_sheets.csv
    content_raw = row.get('content_raw')
    if not content_raw or (isinstance(content_raw, float) and pd.isna(content_raw)) or str(content_raw).strip() == '':
        content_raw = get_lisa_content(str(row.get('id', '')))
    lisa_data = parser_lisa_sheet(str(content_raw))

    # Construire la carte complète
    card = {
        "item_id": f"MCQ-{index+1:06d}",
        "source_material": str(row.get('id', 'LISA Sheet')),
        "generator_info": model,
        "output_format": "JSON",
        "mcq_question": str(row.get('question', '')),
        "question_comment": str(row.get('question_comment', '') or ''),
        "options": {
            "A": str(row.get('option_a', '')),
            "B": str(row.get('option_b', '')),
            "C": str(row.get('option_c', '')),
            "D": str(row.get('option_d', ''))
        },
        "option_comments": {
            "A": str(row.get('option_a_comment', '') or ''),
            "B": str(row.get('option_b_comment', '') or ''),
            "C": str(row.get('option_c_comment', '') or ''),
            "D": str(row.get('option_d_comment', '') or '')
        },
        "correct_option": str(row.get('correct_option', 'A')).upper(),
        "section_a_checks": section_a_checks,
        "section_b_checks": section_b_checks,
        "decision_policy": "Accept if all hard constraints pass and no critical AI-judge dimension fails.",
        "final_decision": "ACCEPT" if all(check['result'] == "PASS" for check in section_a_checks + section_b_checks) else "REVISE",
        "audit_trail": "Judge model: Automated, consistency: evaluated from CSV metrics.",
        "lisa_texte_brut": str(content_raw),
        "lisa_metadata": {
            "identifiant": lisa_data['identifiant'],
            "rang": lisa_data['rang'],
            "rubrique": lisa_data['rubrique'],
            "intitule": lisa_data['intitule'],
            "item_parent": lisa_data['item_parent'],
            "description": lisa_data['description'],
            "contenu": lisa_data['contenu']
        }
    }

    return card


# ============================================================================
# ROUTES API
# ============================================================================

@router.get("/mcq/assigned")
async def get_assigned_mcqs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Retourne la liste des MCQ assignes a l'utilisateur avec leur modele.
    Format: { assignments: [{mcq_id, model}, ...], total, user }
    """
    try:
        db_assignments = db.query(DBMCQAssignment).filter(
            DBMCQAssignment.user_id == current_user.id
        ).all()

        # Si aucune assignation, en creer par defaut
        if not db_assignments:
            print(f"Aucune assignation DB pour {current_user.username}, creation par defaut de 10 MCQs")
            assignment_data = assign_mcqs_to_user(current_user.username, 10)
            model = assignment_data["model"]
            mcq_ids = assignment_data["mcq_ids"]

            for mcq_id in mcq_ids:
                db_assignment = DBMCQAssignment(
                    user_id=current_user.id,
                    mcq_id=mcq_id,
                    model=model,
                    status="pending"
                )
                db.add(db_assignment)
            db.commit()

            assignments_list = [{"mcq_id": mid, "model": model} for mid in mcq_ids]
        else:
            # Les MCQ custom (générés par l'utilisateur) passent en tête de file
            db_assignments.sort(key=lambda a: (0 if a.model == "custom" else 1, a.id))
            assignments_list = [
                {"mcq_id": a.mcq_id, "model": a.model}
                for a in db_assignments
            ]

        return {
            "assignments": assignments_list,
            "total": len(assignments_list),
            "user": current_user.username
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading MCQ list: {str(e)}")


@router.get("/mcq/batch")
async def get_mcqs_batch(
    start: int = 0,
    limit: int = 10,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Retourne un batch de MCQ (pour pagination)
    """
    try:
        if not CSV_PATH.exists():
            raise HTTPException(status_code=404, detail=f"CSV file not found at {CSV_PATH}")

        df = pd.read_csv(CSV_PATH, engine='python', quotechar='"', on_bad_lines='skip')

        # Limiter le batch
        end = min(start + limit, len(df))

        cards = []
        model = "qwen3_4b_pdapt_slerp"

        for idx in range(start, end):
            row = df.iloc[idx]
            card = build_mcq_card_from_row(row, idx, model)
            cards.append(card)

        return {
            "cards": cards,
            "start": start,
            "end": end,
            "total": len(df),
            "has_more": end < len(df)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading MCQ batch: {str(e)}")


@router.get("/mcq/assignments")
async def get_all_assignments(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Récupérer toutes les assignations
    Nécessite les droits admin
    """
    # Vérifier que current_user est admin
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        assignments = load_assignments()

        return {
            "status": "success",
            "data": assignments,
            "total_users": len(assignments)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading assignments: {str(e)}")


@router.get("/mcq/{mcq_id}")
async def get_mcq_by_id(
    mcq_id: str,
    model: str = Query(..., description="Model name"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> MCQCard:
    """
    Retourne une carte MCQ complete avec toutes ses donnees.
    Le modele est passe en query param pour distinguer les MCQs de modeles differents.
    """
    try:
        # Verifier l'assignation avec le modele specifique
        assignment = db.query(DBMCQAssignment).filter(
            DBMCQAssignment.user_id == current_user.id,
            DBMCQAssignment.mcq_id == mcq_id,
            DBMCQAssignment.model == model
        ).first()

        if not assignment:
            raise HTTPException(status_code=403, detail="MCQ not assigned to this user")

        model = assignment.model

        # --- Cas spécial : MCQ généré depuis contenu personnalisé ---
        if model == "custom":
            # mcq_id format: "CSTM-XXXXXXX-1" or "CSTM-XXXXXXX-2"
            parts = mcq_id.rsplit("-", 1)
            if len(parts) != 2:
                raise HTTPException(status_code=400, detail="Invalid custom MCQ ID format")
            job_id = parts[0]
            mcq_index = int(parts[1]) - 1

            custom_mcqs_dir = Path(__file__).parent.parent.parent / "data" / "custom_mcqs"
            job_file = custom_mcqs_dir / f"{job_id}.json"
            if not job_file.exists():
                raise HTTPException(status_code=404, detail="Custom MCQ data not found")

            with open(job_file, "r", encoding="utf-8") as f:
                mock_mcqs = json.load(f)

            if mcq_index < 0 or mcq_index >= len(mock_mcqs):
                raise HTTPException(status_code=404, detail="Custom MCQ index out of range")

            return MCQCard(**mock_mcqs[mcq_index])

        # --- Cas standard : MCQ depuis CSV de modèle ---

        # Obtenir le chemin du CSV pour ce modèle
        csv_path = get_csv_path_for_model(model)

        # Extraire l'index depuis l'ID (MCQ-000001 -> 0)
        try:
            index = int(mcq_id.split('-')[1]) - 1
        except:
            raise HTTPException(status_code=400, detail="Invalid MCQ ID format")

        # Charger le CSV
        df = pd.read_csv(csv_path, engine='python', quotechar='"', on_bad_lines='skip')

        if index < 0 or index >= len(df):
            raise HTTPException(status_code=404, detail="MCQ not found")

        # Récupérer la ligne
        row = df.iloc[index]

        # Construire la carte MCQ
        card = build_mcq_card_from_row(row, index, model)

        return MCQCard(**card)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading MCQ: {str(e)}")


@router.post("/mcq/{mcq_id}/validate")
async def validate_mcq(
    mcq_id: str,
    validation: ValidationRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Enregistre la validation d'un MCQ par l'utilisateur
    """
    try:
        # TODO: Implémenter la sauvegarde des validations dans un fichier JSON ou base de données
        # Pour l'instant, juste retourner un succès

        validation_data = {
            "mcq_id": mcq_id,
            "user": current_user.username,
            "section_a_checks": validation.section_a_checks,
            "section_b_checks": validation.section_b_checks,
            "human_decision": validation.human_decision,
            "human_feedback": validation.human_feedback,
            "timestamp": pd.Timestamp.now().isoformat()
        }

        # TODO: Sauvegarder dans un fichier ou base de données
        # Exemple: sauvegarder dans validations/{user}/{mcq_id}.json

        return {
            "status": "success",
            "message": "Validation saved successfully",
            "data": validation_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving validation: {str(e)}")


@router.post("/mcq/assign")
async def create_assignment(
    assignment: AssignmentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Créer une assignation de MCQs pour un utilisateur
    Nécessite les droits admin. Sauvegarde dans le JSON ET la base de données.
    """
    # Vérifier que current_user est admin
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        model = assignment.model or "qwen3_4b_pdapt_slerp"

        # Auto-repair: synchroniser le tracker avec la DB
        max_assignment = db.query(DBMCQAssignment).filter(
            DBMCQAssignment.model == model
        ).order_by(DBMCQAssignment.mcq_id.desc()).first()

        if max_assignment:
            try:
                max_db_index = int(max_assignment.mcq_id.split('-')[1]) - 1
            except (ValueError, IndexError):
                max_db_index = -1

            tracker = load_global_tracker()
            if model not in tracker:
                tracker[model] = {"last_assigned_index": -1, "total_available": 0, "assigned_count": 0}
            if max_db_index > tracker[model]["last_assigned_index"]:
                print(f"🔧 Auto-repair tracker for '{model}': {tracker[model]['last_assigned_index']} → {max_db_index}")
                tracker[model]["last_assigned_index"] = max_db_index
                tracker[model]["assigned_count"] = max_db_index + 1
                save_global_tracker(tracker)

        # Créer l'assignation (JSON + tracker)
        assignment_data = assign_mcqs_to_user(assignment.username, assignment.count, model)

        # Résoudre le user_id depuis le username
        from ..utils.dependencies import get_user_by_username
        user_data = get_user_by_username(assignment.username)
        if not user_data:
            raise HTTPException(status_code=404, detail=f"User '{assignment.username}' not found")

        # Sauvegarder aussi en base de données
        for mcq_id in assignment_data["mcq_ids"]:
            existing = db.query(DBMCQAssignment).filter(
                DBMCQAssignment.user_id == user_data["id"],
                DBMCQAssignment.mcq_id == mcq_id,
                DBMCQAssignment.model == assignment_data["model"]
            ).first()
            if not existing:
                db_assignment = DBMCQAssignment(
                    user_id=user_data["id"],
                    mcq_id=mcq_id,
                    model=assignment_data["model"],
                    status="pending"
                )
                db.add(db_assignment)
        db.commit()

        return {
            "status": "success",
            "message": f"Assigned {len(assignment_data['mcq_ids'])} MCQs from model '{assignment.model}' to {assignment.username}",
            "data": {
                "username": assignment.username,
                "model": assignment_data["model"],
                "mcq_ids": assignment_data["mcq_ids"],
                "count": len(assignment_data["mcq_ids"])
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating assignment: {str(e)}")


@router.get("/mcq-models")
async def get_available_models() -> List[Dict[str, Any]]:
    """
    Retourne la liste des modèles disponibles avec le nombre de MCQs disponibles (non assignés)
    Format: [{ "model": "qwen3_8b_pdapt_slerp", "count": 320 }, ...]
    Note: Route renamed from /mcq/models to /mcq-models to avoid conflict with /mcq/{mcq_id}
    """
    try:
        models_info = []

        # Scanner le dossier dataset_with_quality pour trouver les fichiers CSV
        if not CSV_DIR.exists():
            print(f"⚠️ CSV directory not found: {CSV_DIR}")
            return []

        # Charger le tracker global pour connaître le nombre de MCQs déjà assignés
        global_tracker = load_global_tracker()

        # Parcourir tous les fichiers CSV dans le dossier
        for csv_file in CSV_DIR.glob("*.csv"):
            model_name = csv_file.stem  # Nom du fichier sans .csv

            try:
                # Lire le CSV pour obtenir le nombre total de MCQs
                df = pd.read_csv(csv_file, engine='python', quotechar='"', on_bad_lines='skip')
                total_mcqs = len(df)

                # Calculer le nombre de MCQs disponibles (non assignés)
                if model_name in global_tracker:
                    last_assigned_index = global_tracker[model_name]["last_assigned_index"]
                    available_count = total_mcqs - (last_assigned_index + 1)
                else:
                    # Aucun MCQ assigné pour ce modèle
                    available_count = total_mcqs

                models_info.append({
                    "model": model_name,
                    "count": total_mcqs,  # Nombre total de MCQs dans le CSV
                    "available": max(0, available_count)  # Non encore assignés
                })

            except Exception as e:
                print(f"⚠️ Error reading CSV for model '{model_name}': {e}")
                continue

        # Trier par nom de modèle
        models_info.sort(key=lambda x: x["model"])

        print(f"📊 Found {len(models_info)} models in {CSV_DIR}")
        return models_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")
