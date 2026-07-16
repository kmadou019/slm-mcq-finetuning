# MCQ Evaluation Backend (FastAPI)

Backend API pour l'application d'évaluation de MCQ.

## 🚀 Installation

```bash
# Installer les dépendances
pip install -r requirements.txt
```

## 🔧 Configuration — fichier `.env`

Créer un fichier `.env` à la racine de `src/page/backend/` (jamais commité) :

```env
# ── Sécurité JWT (obligatoire) ─────────────────────────────────────────────
# Générer avec : python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY=<votre_clé_secrète_aléatoire>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=480

# ── Base de données ────────────────────────────────────────────────────────
DATABASE_URL=sqlite:///./mcq_evaluation.db

# ── CORS ──────────────────────────────────────────────────────────────────
# URLs du frontend autorisées (séparées par des virgules)
ALLOWED_ORIGINS=http://localhost:4200,http://localhost:4201

# ── Rate limiting ──────────────────────────────────────────────────────────
RATE_LIMIT_LOGIN=5/minute
RATE_LIMIT_GENERAL=100/minute

# ── Environnement ─────────────────────────────────────────────────────────
ENVIRONMENT=development   # ou "production"

# ── OpenAI (pipeline d'évaluation : disclosure, difficulty, answerability) ─
OPENAI_API_KEY=sk-...

# ── HuggingFace (téléchargement de modèles) ───────────────────────────────
HF_TOKEN=hf_...

# ── Inférence GPU distante ────────────────────────────────────────────────
OLLAMA_HOST=http://localhost:11434
VLLM_HOST=http://localhost:8001

# API OpenAI-compatible (LiteLLM, vLLM, etc.)
OPENAI_COMPATIBLE_URL=http://localhost:4000/v1
OPENAI_COMPATIBLE_KEY=

# Modèle utilisé pour l'extraction de model card (défaut : mistral:7b-instruct)
MODEL_CARD_EXTRACTOR_MODEL=mistral:7b-instruct

# ── GraphDB / LISA (enrichissement du prompt) ─────────────────────────────
GRAPHDB_MCP_URL=https://mcp-graphdb.sides3.network/mcp
GRAPHDB_BEARER_TOKEN=<votre_token>
```

> **Variables obligatoires** : `SECRET_KEY`, `OPENAI_API_KEY` (si pipeline eval utilisé).
> Toutes les autres ont des valeurs par défaut et sont optionnelles.

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
