"""
Authentication routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
from ..models.auth import (
    LoginRequest,
    LoginResponse,
    User,
    MCQSelectionRequest,
    MCQAssignment
)
from ..utils.security import create_access_token
from ..utils.dependencies import get_current_user, get_user_by_username

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
    current_user: User = Depends(get_current_user)
):
    """
    Assign MCQ questions to the current user
    """
    # TODO: In production, fetch real MCQ IDs from database
    # For now, generate mock MCQ IDs
    mcq_ids = [f"MCQ-{i:06d}" for i in range(1, request.count + 1)]

    assignment = MCQAssignment(
        user_id=current_user.id,
        mcq_count=request.count,
        assigned_mcq_ids=mcq_ids,
        assigned_at=datetime.utcnow().isoformat() + "Z",
        status="pending"
    )

    return assignment
