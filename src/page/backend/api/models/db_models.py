"""
Database models using SQLAlchemy
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, CheckConstraint, UniqueConstraint
from sqlalchemy.sql import func
from database import Base


class MCQAssignment(Base):
    """
    Table pour stocker les assignments de MCQ aux utilisateurs
    """
    __tablename__ = "mcq_assignments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    mcq_id = Column(String, nullable=False)
    model = Column(String, nullable=False)
    assigned_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String, default="pending")  # pending, in_progress, completed

    # Contraintes - unique par (user, mcq, model) pour permettre le meme mcq_id de modeles differents
    __table_args__ = (
        UniqueConstraint('user_id', 'mcq_id', 'model', name='unique_user_mcq_model'),
        CheckConstraint(status.in_(['pending', 'in_progress', 'completed']), name='valid_status'),
    )

    def __repr__(self):
        return f"<MCQAssignment(user={self.user_id}, mcq={self.mcq_id}, model={self.model}, status={self.status})>"


class Validation(Base):
    """
    Table pour stocker les validations des MCQ par les évaluateurs
    """
    __tablename__ = "validations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    mcq_id = Column(String, nullable=False)
    model = Column(String, nullable=False)
    decision = Column(String, nullable=False)  # ACCEPT, REJECT
    human_feedback = Column(Text, nullable=True)
    section_a_checks = Column(Text, nullable=True)  # JSON string
    section_b_checks = Column(Text, nullable=True)  # JSON string
    validated_fields = Column(Text, nullable=True)  # JSON string - {"check_description": true/false}
    validation_duration_seconds = Column(Integer, nullable=True)
    mcq_data = Column(Text, nullable=True)  # JSON string - full MCQ content
    validated_at = Column(DateTime(timezone=True), server_default=func.now())

    # Contraintes - unique par (user, mcq, model)
    __table_args__ = (
        UniqueConstraint('user_id', 'mcq_id', 'model', name='unique_user_mcq_model_validation'),
        CheckConstraint(decision.in_(['ACCEPT', 'REJECT']), name='valid_decision'),
    )

    def __repr__(self):
        return f"<Validation(user={self.user_id}, mcq={self.mcq_id}, model={self.model}, decision={self.decision})>"

    def to_dict(self):
        """Convert to dictionary for API responses"""
        import json as _json
        result = {
            "id": self.id,
            "user_id": self.user_id,
            "mcq_id": self.mcq_id,
            "model": self.model,
            "decision": self.decision,
            "human_feedback": self.human_feedback,
            "validated_fields": _json.loads(self.validated_fields) if self.validated_fields else {},
            "validation_duration_seconds": self.validation_duration_seconds,
            "validated_at": self.validated_at.isoformat() if self.validated_at else None
        }
        if self.mcq_data:
            try:
                mcq = _json.loads(self.mcq_data)
                mcq.pop("lisa_metadata", None)
                result["mcq_data"] = mcq
            except (ValueError, TypeError):
                result["mcq_data"] = self.mcq_data
        return result
