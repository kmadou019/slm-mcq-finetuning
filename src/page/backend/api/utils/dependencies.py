"""
FastAPI dependencies
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from ..models.auth import User
from .security import decode_access_token

# Security scheme
security = HTTPBearer()

# Mock database - In production, use a real database
# For demo, we'll have a simple in-memory user store
# TEMPORARY: Using plain passwords for testing (replace with hashed in production)
MOCK_USERS = {
    "admin": {
        "id": "user-001",
        "username": "admin",
        "email": "admin@mcq-eval.com",
        "role": "admin",
        "created_at": "2024-01-01T00:00:00Z",
        "password": "admin123"  # TEMPORARY: plain password for testing
    },
    "evaluator": {
        "id": "user-002",
        "username": "evaluator",
        "email": "evaluator@mcq-eval.com",
        "role": "evaluator",
        "created_at": "2024-01-01T00:00:00Z",
        "password": "eval123"  # TEMPORARY: plain password for testing
    }
}


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Get the current authenticated user from JWT token
    """
    token = credentials.credentials

    # Decode token
    payload = decode_access_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get username from payload
    username: Optional[str] = payload.get("sub")

    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user from mock database
    user_data = MOCK_USERS.get(username)

    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Return User object (without password)
    return User(
        id=user_data["id"],
        username=user_data["username"],
        email=user_data["email"],
        role=user_data["role"],
        created_at=user_data["created_at"]
    )


def get_user_by_username(username: str) -> Optional[dict]:
    """
    Get user by username from mock database
    """
    return MOCK_USERS.get(username)
