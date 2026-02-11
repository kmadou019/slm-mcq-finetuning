"""
MCQ Routes - API endpoints for MCQ evaluation
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
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
DATA_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "data"
CSV_DIR = DATA_DIR / "dataset_with_quality"  # Fichiers avec résultats de qualité
CSV_PATH = CSV_DIR / "qwen3_8b_pdapt_slerp.csv"  # CSV par défaut
ASSIGNMENTS_PATH = DATA_DIR / "assignments.json"  # Assignations par utilisateur
GLOBAL_TRACKER_PATH = DATA_DIR / "global_assignment_tracker.json"  # Tracker global par modèle

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
    options: Dict[str, str]
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


def load_assignments() -> Dict[str, Dict[str, Any]]:
    """
    Charger les assignations depuis le fichier JSON
    Format: {
        "username": {
            "model": "qwen3_4b_pdapt_slerp",
            "mcq_ids": ["MCQ-000001", "MCQ-000002", ...],
            "assigned_at": "2026-02-10T12:00:00"
        }
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


def assign_mcqs_to_user(username: str, count: int, model: str = "qwen3_8b_pdapt_slerp") -> Dict[str, Any]:
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

    # Initialiser le tracker pour ce modèle s'il n'existe pas
    if model not in global_tracker:
        global_tracker[model] = {
            "last_assigned_index": -1,  # -1 signifie qu'aucun MCQ n'a été assigné
            "total_available": total_mcqs,
            "assigned_count": 0
        }

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

    # Charger les assignations existantes
    assignments = load_assignments()

    # Ajouter/mettre à jour l'assignation pour cet utilisateur
    assignments[username] = {
        "model": model,
        "mcq_ids": mcq_ids,
        "assigned_at": pd.Timestamp.now().isoformat()
    }

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
            "result": evaluation_to_pass_warn(row.get('distractor_quality'), True),
            "status": "not_checked",
            "confidence": None,
            "score": "True/False",
            "notes": f"Score: {row.get('distractor_quality', 'N/A')}"
        },
        {
            "check_id": "B4",
            "description": "Difficulty appropriateness",
            "result": evaluation_to_pass_warn(difficulty, "medium"),
            "status": "not_checked",
            "confidence": None,
            "score": "low/med/high",
            "notes": f"Judge: {difficulty_mapping.get(difficulty, 'medium')}"
        }
    ]

    # Parser LISA Sheet
    lisa_data = parser_lisa_sheet(str(row.get('content_raw', '')))

    # Construire la carte complète
    card = {
        "item_id": f"MCQ-{index+1:06d}",
        "source_material": str(row.get('id', 'LISA Sheet')),
        "generator_info": model,
        "output_format": "JSON",
        "mcq_question": str(row.get('question', '')),
        "options": {
            "A": str(row.get('option_a', '')),
            "B": str(row.get('option_b', '')),
            "C": str(row.get('option_c', '')),
            "D": str(row.get('option_d', ''))
        },
        "correct_option": str(row.get('correct_option', 'A')).upper(),
        "section_a_checks": section_a_checks,
        "section_b_checks": section_b_checks,
        "decision_policy": "Accept if all hard constraints pass and no critical AI-judge dimension fails.",
        "final_decision": "ACCEPT" if all(check['result'] == "PASS" for check in section_a_checks + section_b_checks) else "REVISE",
        "audit_trail": "Judge model: Automated, consistency: evaluated from CSV metrics.",
        "lisa_texte_brut": str(row.get('content_raw', '')),
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
    Retourne la liste des IDs de MCQ assignés à l'utilisateur depuis la base de données
    Si aucune assignation n'existe, en crée une par défaut (10 MCQs du modèle par défaut)
    """
    try:
        # Récupérer les assignments depuis la base de données
        db_assignments = db.query(DBMCQAssignment).filter(
            DBMCQAssignment.user_id == current_user.id
        ).all()

        # Si l'utilisateur n'a pas d'assignation, en créer une par défaut
        if not db_assignments:
            print(f"⚠️ Aucune assignation DB pour {current_user.username}, création d'une assignation par défaut de 10 MCQs")
            assignment_data = assign_mcqs_to_user(current_user.username, 10)
            model = assignment_data["model"]
            mcq_ids = assignment_data["mcq_ids"]

            # Sauvegarder dans la DB
            for mcq_id in mcq_ids:
                db_assignment = DBMCQAssignment(
                    user_id=current_user.id,
                    mcq_id=mcq_id,
                    model=model,
                    status="pending"
                )
                db.add(db_assignment)
            db.commit()
        else:
            # Récupérer les mcq_ids et le modèle depuis la DB
            mcq_ids = [assignment.mcq_id for assignment in db_assignments]
            model = db_assignments[0].model if db_assignments else "qwen3_8b_pdapt_slerp"

        return {
            "mcq_ids": mcq_ids,
            "total": len(mcq_ids),
            "model": model,
            "user": current_user.username
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading MCQ list: {str(e)}")


@router.get("/mcq/{mcq_id}")
async def get_mcq_by_id(mcq_id: str, current_user: User = Depends(get_current_user)) -> MCQCard:
    """
    Retourne une carte MCQ complète avec toutes ses données
    Le modèle est déterminé depuis l'assignation de l'utilisateur
    """
    try:
        # Charger les assignations pour trouver le modèle
        assignments = load_assignments()

        if current_user.username not in assignments:
            raise HTTPException(status_code=404, detail="No assignment found for user")

        user_assignment = assignments[current_user.username]
        model = user_assignment["model"]
        assigned_mcqs = user_assignment["mcq_ids"]

        # Vérifier que le MCQ est bien assigné à cet utilisateur
        if mcq_id not in assigned_mcqs:
            raise HTTPException(status_code=403, detail="MCQ not assigned to this user")

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
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Créer une assignation de MCQs pour un utilisateur
    Nécessite les droits admin (pour l'instant, tous les utilisateurs peuvent le faire)
    """
    try:
        # TODO: Vérifier que current_user est admin
        # if current_user.role != "admin":
        #     raise HTTPException(status_code=403, detail="Admin access required")

        # Créer l'assignation avec le modèle spécifié
        assignment_data = assign_mcqs_to_user(assignment.username, assignment.count, assignment.model)

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


@router.get("/mcq/assignments")
async def get_all_assignments(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Récupérer toutes les assignations
    Nécessite les droits admin (pour l'instant, tous les utilisateurs peuvent le faire)
    """
    try:
        # TODO: Vérifier que current_user est admin
        assignments = load_assignments()

        return {
            "status": "success",
            "data": assignments,
            "total_users": len(assignments)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading assignments: {str(e)}")


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
                    "count": max(0, available_count)  # Ne pas retourner de valeurs négatives
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
