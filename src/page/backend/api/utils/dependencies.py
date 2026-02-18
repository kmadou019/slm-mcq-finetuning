"""
FastAPI dependencies
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from pathlib import Path
import json
from ..models.auth import User
from .security import decode_access_token

# Security scheme
security = HTTPBearer()

# Users database file path
USERS_DB_PATH = Path(__file__).parent.parent.parent / "data" / "users.json"

# Initialize users database
def init_users_db():
    """Load users database file. It must be created first with generate_passwords.py"""
    if not USERS_DB_PATH.exists():
        print(f"⚠️ users.json not found at {USERS_DB_PATH}")
        print(f"   Run: cd backend && python generate_passwords.py")
        return {}
    return load_users_db()

def load_users_db():
    """Load users from JSON file into dictionary"""
    try:
        with open(USERS_DB_PATH, 'r') as f:
            users_list = json.load(f)
        # Convert list to dict with username as key
        return {user["username"]: user for user in users_list}
    except Exception as e:
        print(f"⚠️ Error loading users DB: {e}")
        # Return empty dict - init_users_db will be called to create the file
        return {}

# Load users database on module import
USERS_DB = init_users_db()


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
            detail="Invalid authentication credentials",  # Generic message
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate token expiration (jose library does this automatically)
    # Check for required fields
    username: Optional[str] = payload.get("sub")
    exp = payload.get("exp")
    role: Optional[str] = payload.get("role")

    if username is None or exp is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user from users database
    user_data = USERS_DB.get(username)

    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",  # Generic message
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Return User object (without password)
    # Role from JWT takes priority over file (prevents file tampering)
    return User(
        id=user_data["id"],
        username=user_data["username"],
        email=user_data["email"],
        role=role or user_data.get("role", "evaluator"),
        created_at=user_data["created_at"]
    )


def get_user_by_username(username: str) -> Optional[dict]:
    """
    Get user by username from users database
    """
    return USERS_DB.get(username)


def save_users_db():
    """Save USERS_DB to the JSON file"""
    USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_DB_PATH, 'w') as f:
        json.dump(list(USERS_DB.values()), f, indent=2)
