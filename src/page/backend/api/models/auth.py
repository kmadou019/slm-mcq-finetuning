"""
Pydantic models for authentication
"""
from pydantic import BaseModel, EmailStr
from typing import Optional, Literal
from datetime import datetime


class User(BaseModel):
    """User model"""
    id: str
    username: str
    email: str
    role: Literal['evaluator', 'admin'] = 'evaluator'
    created_at: str


class UserInDB(User):
    """User model with hashed password (for database)"""
    hashed_password: str


class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response model"""
    access_token: str
    token_type: str = "bearer"
    user: User


class MCQSelectionRequest(BaseModel):
    """MCQ selection request"""
    count: int
    model: Optional[str] = "qwen3_4b_pdapt_slerp"  # Modèle par défaut


class MCQAssignment(BaseModel):
    """MCQ assignment model"""
    user_id: str
    mcq_count: int
    assigned_mcq_ids: list[str]
    assigned_at: str
    status: Literal['pending', 'in_progress', 'completed'] = 'pending'
