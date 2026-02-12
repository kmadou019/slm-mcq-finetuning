"""
FastAPI dependencies
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from pathlib import Path
import json
from ..models.auth import User
from .security import decode_access_token, hash_password

# Security scheme
security = HTTPBearer()

# Users database file path
USERS_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "users.json"

# Default users for initialization (passwords will be hashed during init)
DEFAULT_USERS_PLAIN = [
    {
        "id": "user-001",
        "username": "admin",
        "email": "admin@mcq-eval.com",
        "role": "admin",
        "created_at": "2024-01-01T00:00:00Z",
        "plain_password": "admin123"  # Will be hashed during init
    },
    {
        "id": "user-002",
        "username": "evaluator",
        "email": "evaluator@mcq-eval.com",
        "role": "evaluator",
        "created_at": "2024-01-01T00:00:00Z",
        "plain_password": "eval123"  # Will be hashed during init
    }
]

# Initialize users database
def init_users_db():
    """Initialize users database file if it doesn't exist"""
    if not USERS_DB_PATH.exists():
        USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Hash passwords before saving
        hashed_users = []
        for user in DEFAULT_USERS_PLAIN:
            hashed_user = user.copy()
            plain_pwd = hashed_user.pop("plain_password")
            hashed_user["hashed_password"] = hash_password(plain_pwd)
            hashed_users.append(hashed_user)

        with open(USERS_DB_PATH, 'w') as f:
            json.dump(hashed_users, f, indent=2)
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
