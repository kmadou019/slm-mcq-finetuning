# MCQ Evaluation Backend (FastAPI)

Backend API pour l'application d'évaluation de MCQ.

## 🚀 Installation

```bash
# Installer les dépendances
pip install -r requirements.txt
```

## 🏃 Lancement

```bash
# Lancer le serveur de développement
python main.py

# Ou avec uvicorn directement
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

L'API sera accessible sur **http://localhost:8000**

## 📚 Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔐 Utilisateurs de test

### Admin
- **Username**: `admin`
- **Password**: `admin123`

### Evaluator
- **Username**: `evaluator`
- **Password**: `eval123`

## 🔌 Endpoints

### Authentication

- `POST /api/auth/login` - Connexion
- `POST /api/auth/logout` - Déconnexion
- `GET /api/auth/me` - Utilisateur actuel
- `POST /api/auth/assign-mcq` - Assigner des MCQ

### Health

- `GET /` - Root endpoint
- `GET /health` - Health check

## 🛠️ Structure

```
backend/
├── main.py                 # Point d'entrée FastAPI
├── requirements.txt        # Dépendances Python
├── api/
│   ├── routes/
│   │   └── auth.py        # Routes authentification
│   ├── models/
│   │   └── auth.py        # Modèles Pydantic
│   └── utils/
│       ├── security.py    # JWT et hashing
│       └── dependencies.py # Dependencies FastAPI
└── card.py                # Code existant
```

## 📝 Notes

- Les utilisateurs sont stockés en mémoire (MOCK_USERS)
- En production, utiliser une vraie base de données
- Changer la SECRET_KEY dans security.py pour la production
- Les tokens JWT expirent après 8 heures
