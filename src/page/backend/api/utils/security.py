"""
Security utilities for JWT and password hashing
TEMPORARY: Simplified version without external dependencies for testing
"""
from datetime import datetime, timedelta
from typing import Optional
import json
import base64

# Configuration
SECRET_KEY = "your-secret-key-change-this-in-production"  # TODO: Move to environment variable
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a simple token (TEMPORARY: not real JWT for testing)
    TODO: Replace with proper JWT in production
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire.isoformat()})

    # Simple base64 encoding (NOT SECURE - for testing only)
    token_string = json.dumps(to_encode)
    encoded = base64.b64encode(token_string.encode()).decode()

    return encoded


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decode the simple token (TEMPORARY: not real JWT verification)
    TODO: Replace with proper JWT decoding in production
    """
    try:
        decoded_bytes = base64.b64decode(token.encode())
        payload = json.loads(decoded_bytes.decode())
        return payload
    except Exception:
        return None
