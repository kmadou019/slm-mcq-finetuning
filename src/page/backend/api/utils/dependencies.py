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
USERS_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "users.json"

# Default users for initialization
DEFAULT_USERS = [
    {
        "id": "user-001",
        "username": "admin",
        "email": "admin@mcq-eval.com",
        "role": "admin",
        "created_at": "2024-01-01T00:00:00Z",
        "password": "admin123"  # TEMPORARY: plain password for testing
    },
    {
        "id": "user-002",
        "username": "evaluator",
        "email": "evaluator@mcq-eval.com",
        "role": "evaluator",
        "created_at": "2024-01-01T00:00:00Z",
        "password": "eval123"  # TEMPORARY: plain password for testing
    }
]

# Initialize users database
def init_users_db():
    """Initialize users database file if it doesn't exist"""
    if not USERS_DB_PATH.exists():
        USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(USERS_DB_PATH, 'w') as f:
            json.dump(DEFAULT_USERS, f, indent=2)
        print(f"✅ Users database initialized at {USERS_DB_PATH}")
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
        # Return default users if file can't be loaded
        return {user["username"]: user for user in DEFAULT_USERS}

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

    # Get user from users database
    user_data = USERS_DB.get(username)

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
    Get user by username from users database
    """
    return USERS_DB.get(username)
