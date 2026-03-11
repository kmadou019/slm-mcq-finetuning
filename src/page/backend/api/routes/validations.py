"""
API routes for MCQ validations
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
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
    Recuperer les statistiques de validation pour l'utilisateur connecte
    """
    validations = db.query(
        Validation.decision,
        func.count(Validation.id).label('count')
    ).filter(
        Validation.user_id == current_user.id
    ).group_by(Validation.decision).all()

    stats_dict = {decision: count for decision, count in validations}

    total_assigned = db.query(func.count(MCQAssignment.id)).filter(
        MCQAssignment.user_id == current_user.id
    ).scalar() or 0

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
    Recuperer toutes les validations de l'utilisateur connecte.
    Cle composite mcq_id::model pour distinguer les MCQs de modeles differents.
    """
    validations = db.query(Validation).filter(
        Validation.user_id == current_user.id
    ).order_by(Validation.validated_at.desc()).all()

    # Cle composite mcq_id::model
    validations_dict = {}
    for val in validations:
        key = f"{val.mcq_id}::{val.model}"
        validations_dict[key] = {
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
    Recuperer l'historique des validations (liste chronologique)
    """
    validations = db.query(Validation).filter(
        Validation.user_id == current_user.id
    ).order_by(Validation.validated_at.desc()).limit(limit).all()

    return [val.to_dict() for val in validations]


@router.post("/{mcq_id}/validate")
async def create_validation(
    mcq_id: str,
    validation_data: Dict[str, Any],
    model: str = Query(..., description="Model name"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Creer ou mettre a jour une validation pour un MCQ.
    Le modele est passe en query param pour identifier l'assignation.
    """
    # Verifier si le MCQ est assigne a l'utilisateur pour ce modele
    assignment = db.query(MCQAssignment).filter(
        MCQAssignment.user_id == current_user.id,
        MCQAssignment.mcq_id == mcq_id,
        MCQAssignment.model == model
    ).first()

    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"MCQ {mcq_id} (model {model}) not assigned to this user"
        )

    # Verifier si une validation existe deja pour ce (user, mcq, model)
    existing_validation = db.query(Validation).filter(
        Validation.user_id == current_user.id,
        Validation.mcq_id == mcq_id,
        Validation.model == model
    ).first()

    mcq_data_json = None
    if validation_data.get("mcq_data"):
        mcq_data_json = json.dumps(validation_data["mcq_data"], ensure_ascii=False)

    # Build validated_fields from section checks if not provided explicitly
    validated_fields = validation_data.get("validated_fields")
    if not validated_fields:
        validated_fields = {}
        for check in validation_data.get("section_a_checks", []):
            desc = check.get("description", check.get("check_id", "unknown"))
            validated_fields[desc] = check.get("status") == "validated"
        for check in validation_data.get("section_b_checks", []):
            desc = check.get("description", check.get("check_id", "unknown"))
            validated_fields[desc] = check.get("status") == "validated"
    validated_fields_json = json.dumps(validated_fields, ensure_ascii=False) if validated_fields else None

    content_raw = validation_data.get("content_raw") or None

    if existing_validation:
        existing_validation.decision = validation_data.get("human_decision", "REJECT")
        existing_validation.human_feedback = validation_data.get("human_feedback", "")
        existing_validation.section_a_checks = json.dumps(validation_data.get("section_a_checks", []))
        existing_validation.section_b_checks = json.dumps(validation_data.get("section_b_checks", []))
        existing_validation.validated_fields = validated_fields_json
        existing_validation.validation_duration_seconds = validation_data.get("validation_duration_seconds")
        if mcq_data_json:
            existing_validation.mcq_data = mcq_data_json
        if content_raw:
            existing_validation.content_raw = content_raw
        existing_validation.validated_at = func.now()

        db.commit()
        db.refresh(existing_validation)

        return {
            "status": "updated",
            "message": "Validation updated successfully",
            "data": existing_validation.to_dict()
        }
    else:
        new_validation = Validation(
            user_id=current_user.id,
            mcq_id=mcq_id,
            model=model,
            decision=validation_data.get("human_decision", "REJECT"),
            human_feedback=validation_data.get("human_feedback", ""),
            section_a_checks=json.dumps(validation_data.get("section_a_checks", [])),
            section_b_checks=json.dumps(validation_data.get("section_b_checks", [])),
            validated_fields=validated_fields_json,
            validation_duration_seconds=validation_data.get("validation_duration_seconds"),
            mcq_data=mcq_data_json,
            content_raw=content_raw
        )

        db.add(new_validation)
        db.commit()
        db.refresh(new_validation)

        assignment.status = "completed"
        db.commit()

        return {
            "status": "created",
            "message": "Validation created successfully",
            "data": new_validation.to_dict()
        }
