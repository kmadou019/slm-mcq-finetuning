"""
Authentication routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from datetime import datetime
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address
from ..models.auth import (
    LoginRequest,
    LoginResponse,
    User,
    MCQSelectionRequest,
    MCQAssignment
)
from ..models.db_models import MCQAssignment as DBMCQAssignment
from ..utils.security import create_access_token, verify_password
from ..utils.dependencies import get_current_user, get_user_by_username
from database import get_db

router = APIRouter(prefix="/auth", tags=["authentication"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/login", response_model=LoginResponse)
@limiter.limit("5/minute")
async def login(request: Request, credentials: LoginRequest):
    """
    Login endpoint - Authenticate user and return JWT token
    Rate limited to 5 attempts per minute
    """
    # Get user from database
    user_data = get_user_by_username(credentials.username)

    # Generic error - don't reveal if user exists
    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify password with bcrypt
    if not verify_password(credentials.password, user_data.get("hashed_password", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create JWT access token
    access_token = create_access_token(data={"sub": user_data["username"], "role": user_data["role"]})

    # Create User object (without password)
    user = User(
        id=user_data["id"],
        username=user_data["username"],
        email=user_data["email"],
        role=user_data["role"],
        created_at=user_data["created_at"]
    )

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user=user
    )


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout endpoint - In JWT stateless auth, this is mostly for consistency
    Frontend should remove the token
    """
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user
    """
    return current_user


@router.post("/assign-mcq", response_model=MCQAssignment)
async def assign_mcq(
    request: MCQSelectionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Assign MCQ questions to the current user
    Uses sequential assignment from CSV files
    Saves assignments to database
    """
    from .mcq import assign_mcqs_to_user, load_global_tracker, save_global_tracker

    model = request.model or "qwen3_4b_pdapt_slerp"

    try:
        # Auto-repair: synchroniser le tracker avec la DB
        # Trouver le max MCQ index assigné pour ce modèle en DB (tous users confondus)
        max_assignment = db.query(DBMCQAssignment).filter(
            DBMCQAssignment.model == model
        ).order_by(DBMCQAssignment.mcq_id.desc()).first()

        if max_assignment:
            try:
                max_db_index = int(max_assignment.mcq_id.split('-')[1]) - 1
            except (ValueError, IndexError):
                max_db_index = -1

            # Mettre à jour le tracker si la DB est en avance
            tracker = load_global_tracker()
            if model not in tracker:
                tracker[model] = {"last_assigned_index": -1, "total_available": 0, "assigned_count": 0}
            if max_db_index > tracker[model]["last_assigned_index"]:
                print(f"🔧 Auto-repair tracker for '{model}': {tracker[model]['last_assigned_index']} → {max_db_index}")
                tracker[model]["last_assigned_index"] = max_db_index
                tracker[model]["assigned_count"] = max_db_index + 1
                save_global_tracker(tracker)

        # Assigner (tracker maintenant à jour)
        result = assign_mcqs_to_user(
            username=current_user.username,
            count=request.count,
            model=model
        )

        # Sauvegarder en DB
        for mcq_id in result["mcq_ids"]:
            existing = db.query(DBMCQAssignment).filter(
                DBMCQAssignment.user_id == current_user.id,
                DBMCQAssignment.mcq_id == mcq_id
            ).first()

            if not existing:
                db_assignment = DBMCQAssignment(
                    user_id=current_user.id,
                    mcq_id=mcq_id,
                    model=model,
                    status="pending"
                )
                db.add(db_assignment)

        db.commit()

        assignment = MCQAssignment(
            user_id=current_user.id,
            mcq_count=len(result["mcq_ids"]),
            assigned_mcq_ids=result["mcq_ids"],
            assigned_at=datetime.utcnow().isoformat() + "Z",
            status="pending"
        )

        return assignment

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error assigning MCQs: {str(e)}"
        )
