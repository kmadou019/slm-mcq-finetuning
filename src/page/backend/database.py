"""
Database configuration for SQLite with SQLAlchemy
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path

# Déterminer le chemin de la base de données
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATABASE_DIR = BASE_DIR / "data"
DATABASE_DIR.mkdir(exist_ok=True)
DATABASE_PATH = DATABASE_DIR / "mcq_evaluation.db"

# URL de connexion SQLite
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Créer le moteur SQLAlchemy
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},  # Nécessaire pour SQLite
    echo=False  # Mettre True pour voir les requêtes SQL en debug
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base pour les modèles
Base = declarative_base()


def get_db():
    """
    Dependency pour obtenir une session de base de données
    Usage dans FastAPI: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialiser la base de données
    Créer toutes les tables si elles n'existent pas
    """
    from api.models.db_models import Validation, MCQAssignment
    Base.metadata.create_all(bind=engine)
    print(f"✅ Base de données initialisée : {DATABASE_PATH}")
