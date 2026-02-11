"""
API routes for MCQ validations
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any
import json

from database import get_db
from api.models.db_models import Validation, MCQAssignment
from api.models.auth import User
from api.routes.auth import get_current_user

router = APIRouter(prefix="/validations", tags=["validations"])


@router.get("/stats")
async def get_validation_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Récupérer les statistiques de validation pour l'utilisateur connecté
    """
    # Compter toutes les validations par décision
    validations = db.query(
        Validation.decision,
        func.count(Validation.id).label('count')
    ).filter(
        Validation.user_id == current_user.id
    ).group_by(Validation.decision).all()

    # Convertir en dictionnaire
    stats_dict = {decision: count for decision, count in validations}

    # Récupérer le nombre total de MCQ assignés
    total_assigned = db.query(func.count(MCQAssignment.id)).filter(
        MCQAssignment.user_id == current_user.id
    ).scalar() or 0

    # Calculer les stats
    accepted = stats_dict.get('ACCEPT', 0)
    rejected = stats_dict.get('REJECT', 0)
    completed = accepted + rejected
    pending = total_assigned - completed

    return {
        "total": total_assigned,
        "pending": pending,
        "completed": completed,
        "accepted": accepted,
        "rejected": rejected
    }


@router.get("/user")
async def get_user_validations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Récupérer toutes les validations de l'utilisateur connecté
    """
    validations = db.query(Validation).filter(
        Validation.user_id == current_user.id
    ).order_by(Validation.validated_at.desc()).all()

    # Convertir en dictionnaire indexé par mcq_id
    validations_dict = {}
    for val in validations:
        validations_dict[val.mcq_id] = {
            "decision": val.decision,
            "timestamp": val.validated_at.isoformat() if val.validated_at else None,
            "human_feedback": val.human_feedback,
            "section_a_checks": json.loads(val.section_a_checks) if val.section_a_checks else [],
            "section_b_checks": json.loads(val.section_b_checks) if val.section_b_checks else [],
            "validation_duration_seconds": val.validation_duration_seconds
        }

    return {
        "user_id": current_user.id,
        "username": current_user.username,
        "validations": validations_dict,
        "total_validations": len(validations)
    }


@router.get("/history")
async def get_validation_history(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Récupérer l'historique des validations (liste chronologique)
    """
    validations = db.query(Validation).filter(
        Validation.user_id == current_user.id
    ).order_by(Validation.validated_at.desc()).limit(limit).all()

    return [val.to_dict() for val in validations]


@router.post("/{mcq_id}/validate")
async def create_validation(
    mcq_id: str,
    validation_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Créer ou mettre à jour une validation pour un MCQ
    """
    # Vérifier si le MCQ est assigné à l'utilisateur
    assignment = db.query(MCQAssignment).filter(
        MCQAssignment.user_id == current_user.id,
        MCQAssignment.mcq_id == mcq_id
    ).first()

    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"MCQ {mcq_id} not assigned to this user"
        )

    # Vérifier si une validation existe déjà
    existing_validation = db.query(Validation).filter(
        Validation.user_id == current_user.id,
        Validation.mcq_id == mcq_id
    ).first()

    if existing_validation:
        # Mettre à jour
        existing_validation.decision = validation_data.get("human_decision", "REJECT")
        existing_validation.human_feedback = validation_data.get("human_feedback", "")
        existing_validation.section_a_checks = json.dumps(validation_data.get("section_a_checks", []))
        existing_validation.section_b_checks = json.dumps(validation_data.get("section_b_checks", []))
        existing_validation.validation_duration_seconds = validation_data.get("validation_duration_seconds")
        existing_validation.validated_at = func.now()

        db.commit()
        db.refresh(existing_validation)

        return {
            "status": "updated",
            "message": "Validation updated successfully",
            "data": existing_validation.to_dict()
        }
    else:
        # Créer une nouvelle validation
        new_validation = Validation(
            user_id=current_user.id,
            mcq_id=mcq_id,
            decision=validation_data.get("human_decision", "REJECT"),
            human_feedback=validation_data.get("human_feedback", ""),
            section_a_checks=json.dumps(validation_data.get("section_a_checks", [])),
            section_b_checks=json.dumps(validation_data.get("section_b_checks", [])),
            validation_duration_seconds=validation_data.get("validation_duration_seconds")
        )

        db.add(new_validation)
        db.commit()
        db.refresh(new_validation)

        # Mettre à jour le statut de l'assignment
        assignment.status = "completed"
        db.commit()

        return {
            "status": "created",
            "message": "Validation created successfully",
            "data": new_validation.to_dict()
        }
