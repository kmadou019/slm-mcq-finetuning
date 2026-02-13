"""
Admin routes - Endpoints pour le dashboard administrateur
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any
import json
from pathlib import Path
from pydantic import BaseModel

from database import get_db
from api.models.db_models import Validation, MCQAssignment
from api.models.auth import User
from api.routes.auth import get_current_user
from api.routes.mcq import load_global_tracker, save_global_tracker, GLOBAL_TRACKER_PATH
from api.utils.security import hash_password
from api.utils.dependencies import USERS_DB, save_users_db

router = APIRouter(prefix="/admin", tags=["admin"])


# ============================================================================
# GUARDS - Vérifier que l'utilisateur est admin
# ============================================================================

def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Vérifier que l'utilisateur a le rôle admin"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


# ============================================================================
# MODELES PYDANTIC
# ============================================================================

class UserCreate(BaseModel):
    username: str
    password: str
    email: str
    role: str = "evaluator"


class UserUpdate(BaseModel):
    username: str | None = None
    email: str | None = None
    password: str | None = None
    role: str | None = None


# ============================================================================
# STATS GLOBALES
# ============================================================================

@router.get("/stats/global")
async def get_global_stats(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Statistiques globales pour le dashboard admin
    """
    # Nombre total de validations
    total_validations = db.query(func.count(Validation.id)).scalar() or 0

    # Nombre par décision
    stats_by_decision = db.query(
        Validation.decision,
        func.count(Validation.id).label('count')
    ).group_by(Validation.decision).all()

    stats_dict = {decision: count for decision, count in stats_by_decision}
    accepted = stats_dict.get('ACCEPT', 0)
    rejected = stats_dict.get('REJECT', 0)

    # Nombre d'évaluateurs actifs (qui ont au moins une validation)
    active_evaluators = db.query(func.count(func.distinct(Validation.user_id))).scalar() or 0

    # Total MCQ assignés
    total_assignments = db.query(func.count(MCQAssignment.id)).scalar() or 0

    # Taux d'acceptation
    acceptance_rate = (accepted / total_validations * 100) if total_validations > 0 else 0

    return {
        "total_validations": total_validations,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": round(acceptance_rate, 2),
        "active_evaluators": active_evaluators,
        "total_assignments": total_assignments,
        "completion_rate": round((total_validations / total_assignments * 100) if total_assignments > 0 else 0, 2)
    }


@router.get("/stats/by-model")
async def get_stats_by_model(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Statistiques par modèle - inclut tous les modèles CSV + ceux en DB
    """
    import pandas as pd
    from api.routes.mcq import CSV_DIR

    # Collecter tous les noms de modèles (CSV + DB)
    all_models = set()

    # Modèles depuis les fichiers CSV
    csv_totals = {}
    if CSV_DIR.exists():
        for csv_file in CSV_DIR.glob("*.csv"):
            model_name = csv_file.stem
            all_models.add(model_name)
            try:
                df = pd.read_csv(csv_file, engine='python', quotechar='"', on_bad_lines='skip')
                csv_totals[model_name] = len(df)
            except Exception:
                csv_totals[model_name] = 0

    # Modèles depuis la DB (au cas où un CSV aurait été supprimé)
    db_models = db.query(MCQAssignment.model).distinct().all()
    for (model,) in db_models:
        all_models.add(model)

    tracker = load_global_tracker()
    stats_by_model = []

    for model in sorted(all_models):
        # Assignments pour ce modèle
        total_assigned = db.query(func.count(MCQAssignment.id)).filter(
            MCQAssignment.model == model
        ).scalar() or 0

        # Validations pour ce modèle
        mcq_ids_query = db.query(MCQAssignment.mcq_id).filter(
            MCQAssignment.model == model
        ).all()
        mcq_ids = [mcq_id for (mcq_id,) in mcq_ids_query]

        accepted = 0
        rejected = 0
        if mcq_ids:
            validations = db.query(
                Validation.decision,
                func.count(Validation.id).label('count')
            ).filter(
                Validation.mcq_id.in_(mcq_ids)
            ).group_by(Validation.decision).all()
            val_dict = {decision: count for decision, count in validations}
            accepted = val_dict.get('ACCEPT', 0)
            rejected = val_dict.get('REJECT', 0)

        evaluated = accepted + rejected

        # Total disponible: CSV d'abord, sinon tracker
        total_available = csv_totals.get(model, 0)
        if total_available == 0:
            total_available = tracker.get(model, {}).get('total_available', 0)

        tracker_info = tracker.get(model, {})
        last_assigned_index = tracker_info.get('last_assigned_index', -1)
        remaining = max(0, total_available - (last_assigned_index + 1)) if total_available > 0 else 0

        acceptance_rate = (accepted / evaluated * 100) if evaluated > 0 else 0

        stats_by_model.append({
            "model": model,
            "total_available": total_available,
            "total_assigned": total_assigned,
            "evaluated": evaluated,
            "accepted": accepted,
            "rejected": rejected,
            "remaining": remaining,
            "acceptance_rate": round(acceptance_rate, 2),
            "progress_percent": round((evaluated / total_assigned * 100) if total_assigned > 0 else 0, 2)
        })

    return stats_by_model


# ============================================================================
# GESTION UTILISATEURS
# ============================================================================

@router.get("/users")
async def list_users(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Liste tous les utilisateurs avec leurs statistiques
    """

    users_list = []

    for user_data in USERS_DB.values():
        user_id = user_data["id"]

        # Stats de cet utilisateur
        validations_count = db.query(func.count(Validation.id)).filter(
            Validation.user_id == user_id
        ).scalar() or 0

        assignments_count = db.query(func.count(MCQAssignment.id)).filter(
            MCQAssignment.user_id == user_id
        ).scalar() or 0

        # Décisions
        decisions = db.query(
            Validation.decision,
            func.count(Validation.id).label('count')
        ).filter(
            Validation.user_id == user_id
        ).group_by(Validation.decision).all()

        dec_dict = {decision: count for decision, count in decisions}

        users_list.append({
            "id": user_id,
            "username": user_data["username"],
            "email": user_data.get("email", ""),
            "role": user_data.get("role", "evaluator"),
            "created_at": user_data.get("created_at", ""),
            "stats": {
                "assignments": assignments_count,
                "validations": validations_count,
                "accepted": dec_dict.get('ACCEPT', 0),
                "rejected": dec_dict.get('REJECT', 0)
            }
        })

    return users_list


@router.post("/users")
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Créer un nouvel utilisateur
    """
    from datetime import datetime
    import uuid

    # Vérifier si le username existe déjà
    if any(u["username"] == user_data.username for u in USERS_DB.values()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )

    # Créer le nouvel utilisateur
    new_user_id = str(uuid.uuid4())
    new_user = {
        "id": new_user_id,
        "username": user_data.username,
        "hashed_password": hash_password(user_data.password),
        "email": user_data.email,
        "role": user_data.role,
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

    # Ajouter à la DB en mémoire et sauvegarder
    USERS_DB[user_data.username] = new_user
    save_users_db()

    return {
        "status": "success",
        "message": "User created successfully",
        "user": {
            "id": new_user_id,
            "username": user_data.username,
            "email": user_data.email,
            "role": user_data.role
        }
    }


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: User = Depends(require_admin)
) -> Dict[str, str]:
    """
    Supprimer un utilisateur (soft delete - ne supprime pas les validations)
    """
    # Trouver et supprimer l'utilisateur
    user_to_delete = None
    for username, user_data in USERS_DB.items():
        if user_data["id"] == user_id:
            user_to_delete = username
            break

    if not user_to_delete:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Ne pas permettre de supprimer les admins
    if USERS_DB[user_to_delete].get("role") == "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete admin users"
        )

    # Supprimer et sauvegarder
    del USERS_DB[user_to_delete]
    save_users_db()

    return {"status": "success", "message": "User deleted successfully"}


@router.put("/users/{user_id}")
async def update_user(
    user_id: str,
    update_data: UserUpdate,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Modifier un utilisateur (email, role, mot de passe)
    """
    # Trouver l'utilisateur par ID
    target_username = None
    for username, user_data in USERS_DB.items():
        if user_data["id"] == user_id:
            target_username = username
            break

    if not target_username:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    user_data = USERS_DB[target_username]

    # Mettre a jour les champs fournis
    if update_data.email is not None:
        user_data["email"] = update_data.email

    if update_data.role is not None:
        user_data["role"] = update_data.role

    if update_data.password is not None and update_data.password.strip():
        user_data["hashed_password"] = hash_password(update_data.password)

    # Si le username change, mettre a jour la cle du dict
    if update_data.username is not None and update_data.username != target_username:
        if update_data.username in USERS_DB:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        user_data["username"] = update_data.username
        del USERS_DB[target_username]
        USERS_DB[update_data.username] = user_data

    save_users_db()

    return {
        "status": "success",
        "message": "User updated successfully",
        "user": {
            "id": user_data["id"],
            "username": user_data["username"],
            "email": user_data.get("email", ""),
            "role": user_data.get("role", "evaluator")
        }
    }


# ============================================================================
# GESTION TRACKER
# ============================================================================

@router.get("/tracker")
async def get_tracker(
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Récupérer le tracker global
    """
    tracker = load_global_tracker()

    return {
        "tracker": tracker,
        "path": str(GLOBAL_TRACKER_PATH),
        "exists": GLOBAL_TRACKER_PATH.exists()
    }


@router.put("/tracker")
async def update_tracker(
    tracker_data: Dict[str, Any],
    current_user: User = Depends(require_admin)
) -> Dict[str, str]:
    """
    Mettre à jour le tracker global
    """
    save_global_tracker(tracker_data)

    return {
        "status": "success",
        "message": "Tracker updated successfully"
    }


@router.post("/tracker/reset/{model}")
async def reset_tracker_for_model(
    model: str,
    current_user: User = Depends(require_admin)
) -> Dict[str, str]:
    """
    Réinitialiser le tracker pour un modèle spécifique
    """
    tracker = load_global_tracker()

    if model in tracker:
        tracker[model] = {
            "last_assigned_index": -1,
            "total_available": tracker[model].get("total_available", 0),
            "assigned_count": 0
        }
        save_global_tracker(tracker)
        return {"status": "success", "message": f"Tracker reset for model {model}"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model} not found in tracker"
        )


# ============================================================================
# EXPORT DONNÉES
# ============================================================================

@router.get("/export/validations")
async def export_validations(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Exporter toutes les validations (format JSON)
    """
    validations = db.query(Validation).all()

    return [val.to_dict() for val in validations]


@router.get("/export/stats")
async def export_stats(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Exporter un rapport complet des statistiques
    """
    global_stats = await get_global_stats(current_user, db)
    model_stats = await get_stats_by_model(current_user, db)
    users = await list_users(current_user, db)

    return {
        "generated_at": Path(__file__).stat().st_mtime,
        "global_stats": global_stats,
        "stats_by_model": model_stats,
        "users": users
    }
