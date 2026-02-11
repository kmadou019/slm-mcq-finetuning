"""
Authentication routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
from sqlalchemy.orm import Session
from ..models.auth import (
    LoginRequest,
    LoginResponse,
    User,
    MCQSelectionRequest,
    MCQAssignment
)
from ..models.db_models import MCQAssignment as DBMCQAssignment
from ..utils.security import create_access_token
from ..utils.dependencies import get_current_user, get_user_by_username
from database import get_db

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=LoginResponse)
async def login(credentials: LoginRequest):
    """
    Login endpoint - Authenticate user and return JWT token
    """
    # Get user from database
    user_data = get_user_by_username(credentials.username)

    # Check if user exists
    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify password (TEMPORARY: plain text comparison for testing)
    # TODO: Replace with verify_password(credentials.password, user_data["hashed_password"]) in production
    if credentials.password != user_data.get("password"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token = create_access_token(data={"sub": user_data["username"]})

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
    # Import the assignment function from mcq routes
    from .mcq import assign_mcqs_to_user

    try:
        # Utiliser la fonction d'assignation séquentielle
        result = assign_mcqs_to_user(
            username=current_user.username,
            count=request.count,
            model=request.model or "qwen3_8b_pdapt_slerp"
        )

        # Sauvegarder les assignments dans la base de données
        for mcq_id in result["mcq_ids"]:
            # Vérifier si l'assignment existe déjà
            existing = db.query(DBMCQAssignment).filter(
                DBMCQAssignment.user_id == current_user.id,
                DBMCQAssignment.mcq_id == mcq_id
            ).first()

            if not existing:
                # Créer un nouvel assignment
                db_assignment = DBMCQAssignment(
                    user_id=current_user.id,
                    mcq_id=mcq_id,
                    model=request.model or "qwen3_8b_pdapt_slerp",
                    status="pending"
                )
                db.add(db_assignment)

        # Commit tous les assignments
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
