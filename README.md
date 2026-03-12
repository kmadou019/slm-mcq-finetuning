# MCQ Generation & Evaluation Pipeline

Plateforme de génération automatique de QCM médicaux à partir des fiches LISA (référentiel national français), avec évaluation qualité automatisée et validation humaine par des experts.

---

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Métriques d'évaluation](#métriques-dévaluation)
- [Prise en main rapide](#prise-en-main-rapide)
- [Structure du projet](#structure-du-projet)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Génération de QCM personnalisés](#génération-de-qcm-personnalisés)
- [Pipeline de fine-tuning](#pipeline-de-fine-tuning)
- [Configuration](#configuration)

---

## Vue d'ensemble

Ce projet implémente une boucle complète **génération → évaluation → validation humaine → fine-tuning** pour des QCM médicaux :

1. **Génération** — Des LLMs (via Ollama/vLLM) génèrent des QCM à partir de fiches LISA
2. **Évaluation automatique** — 9 métriques de qualité sont calculées automatiquement (linguistiques, sémantiques, LLM-as-judge)
3. **Validation humaine** — Des évaluateurs experts valident chaque QCM via une interface web
4. **Fine-tuning** — Les préférences humaines (ACCEPT/REJECT) alimentent un pipeline SFT/DPO

### Modèles supportés

Les QCM générés sont organisés par modèle (`data/mcqs/{model_name}.csv`). Chaque CSV contient les QCM avec leurs métriques pré-calculées.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     FLUX DE DONNÉES                           │
│                                                              │
│  Fiches LISA (CSV)                                           │
│       │                                                      │
│       ▼                                                      │
│  notebooks/generate_mcq.py  ←── LLM (Ollama / vLLM)         │
│       │                                                      │
│       ▼                                                      │
│  data/mcqs/{model}.csv  ←── eval pipeline (9 métriques)     │
│       │                                                      │
│       ▼                                                      │
│  Backend FastAPI (port 8000)                                 │
│       │                                                      │
│       ▼                                                      │
│  Frontend Angular (port 4200)  ←── Évaluateurs experts      │
│       │                                                      │
│       ▼                                                      │
│  SQLite DB (validations)                                     │
│       │                                                      │
│       ▼                                                      │
│  notebooks/dpo_training.ipynb  ←── SFT / DPO fine-tuning    │
└──────────────────────────────────────────────────────────────┘
```

**Stack technique :**

| Composant | Technologie |
|-----------|-------------|
| Backend | FastAPI + SQLAlchemy + SQLite |
| Frontend | Angular 21 (standalone components, signals) |
| Authentification | JWT (HS256) + bcrypt |
| Génération | Ollama (local) / vLLM (GPU distant) |
| Évaluation ML | ModernCamemBERT (embeddings sémantiques) |
| Évaluation LLM | OpenAI GPT-4o (distracteurs, difficulté, disclosure) |

---

## Métriques d'évaluation

### Section A — Format & Structure

| ID | Métrique | Description | Seuil | Outil |
|----|----------|-------------|-------|-------|
| A1 | **Is Question** | Vérifie que le texte est bien une question (point d'interrogation, mots interrogatifs français) | PASS requis | Règles linguistiques |
| A2 | **No Leading Negation** | Détecte si la question commence par une négation (ex: "N'est-il pas..."), ce qui augmente la charge cognitive | Absence de négation | Règles lexicales |
| A3 | **Originality** | Mesure que le LLM n'a pas copié-collé la fiche LISA. Calcule le ratio de trigrammes uniques de la question par rapport aux trigrammes de la fiche. Score élevé = original | ≥ 0.75 | NLTK trigrammes + lemmatisation |
| A4 | **Readability** | Indice de lisibilité Flesch-Kincaid Grade Level. Mesure la complexité syntaxique et lexicale. Un QCM médical doit être complexe mais pas illisible. | ≥ 12 | FK formula |

### Section B — Qualité du Contenu

| ID | Métrique | Description | Seuil | Outil |
|----|----------|-------------|-------|-------|
| B1 | **Relevance** | Similarité cosinus entre les embeddings de la question et de la fiche LISA source. Mesure que la question porte bien sur le bon sujet. | ≥ 0.5 | ModernCamemBERT |
| B2 | **Ambiguity** | Similarité cosinus moyenne entre la bonne réponse et les distracteurs. Un score élevé signifie que les distracteurs sont trop proches de la réponse (ambiguïté). | ≤ 0.65 | ModernCamemBERT |
| B3 | **Distractors Quality** | GPT-4o évalue chaque distracteur sur sa plausibilité (1–5). Vérifie que les mauvaises réponses sont crédibles mais incorrectes. Les seuils varient selon le rang LISA. | Rang A: ≥3 passés, Rang B: ≥4 passés | GPT-4o |
| B4 | **Disclosure** | GPT-4o détecte si le libellé de la question laisse trop clairement deviner la bonne réponse (answer leakage). | False (pas de fuite) | GPT-4o |
| B5 | **Difficulty** | GPT-4o évalue la difficulté du QCM (1=très facile, 5=très difficile). Un QCM trop trivial ou impossible n'a pas de valeur pédagogique. | Moyen (3) idéal | GPT-4o |
| B6 | **Answerability** | Vérifie si un LLM peut répondre correctement depuis le contexte seul (sans distracteurs). Teste la qualité de la question isolément. | Configurable | OpenAI API |

### Différence A3 vs B2

| | A3 — Originality | B2 — Ambiguity |
|---|---|---|
| Ce qui est comparé | Question ↔ Fiche LISA | Bonne réponse ↔ Distracteurs |
| Objectif | Le LLM n'a pas copié | Les distracteurs se distinguent bien |
| Score élevé signifie | Original (bien) | Trop ambigu (mal) |
| Score bas signifie | Copié-collé (mal) | Bien discriminant (bien) |

---

## Prise en main rapide

### Prérequis

- Python 3.11+
- Node.js v20+ (ou v22+)
- Ollama installé et en cours d'exécution (`ollama serve`)
- Clé API OpenAI (pour les métriques GPT-4o)

### 1. Backend

```bash
cd src/page/backend

# Installer les dépendances
pip install -r requirements.txt

# Créer les utilisateurs initiaux
python generate_passwords.py

# Lancer le serveur (uvicorn avec hot-reload)
./main.py
# → http://localhost:8000
# → Swagger UI : http://localhost:8000/docs
```

### 2. Frontend

```bash
cd src/page/frontend/mcq-evalutation

# Installer les dépendances
npm install

# Lancer le serveur de développement
ng serve
# → http://localhost:4200
```

> **Note Node.js :** Si `ng serve` échoue avec une erreur de version, vérifiez que Node.js >= v20 est actif.
> Le CLI Angular (`ng`) est géré via pnpm — assurez-vous que `~/.local/share/pnpm` est dans votre `PATH`.

### 3. Générer des QCM

```bash
# Depuis la racine du projet
python notebooks/generate_mcq.py \
  --model qwen3:8b \
  --lisa data/lisa_sheets.csv \
  --output data/mcqs/qwen3_8b.csv
```

### 4. Calculer les métriques sur un CSV existant

```python
from eval.eval_dataframe import evaluate_dataframe
import pandas as pd

df = pd.read_csv("data/mcqs/mon_modele.csv")
df = evaluate_dataframe(df)
df.to_csv("data/mcqs/mon_modele.csv", index=False)
```

---

## Structure du projet

```
slm-mcq-finetuning/
│
├── notebooks/                          # Scripts & notebooks
│   ├── generate_mcq.py                 # Génération batch (CLI)
│   ├── generate_mcq.ipynb              # Génération (interactif)
│   ├── dpo_training.ipynb              # Fine-tuning SFT/DPO
│   ├── answerability.ipynb             # Évaluation answerability
│   ├── benchmark_nemotron_vs_magistral.py
│   └── reliability_test.py
│
├── data/
│   ├── lisa_sheets.csv                 # Fiches LISA (référentiel)
│   ├── mcqs/                           # Un CSV par modèle
│   │   ├── qwen3_8b.csv
│   │   └── ...
│   ├── custom_mcqs/                    # QCM générés depuis contenu custom
│   │   └── CSTM-XXXXXXX.json
│   ├── mcq_evaluation.db               # SQLite (assignments + validations)
│   ├── users.json                      # Utilisateurs
│   ├── assignments.json                # Assignations utilisateurs
│   └── global_assignment_tracker.json  # Suivi séquentiel des assignations
│
└── src/page/
    ├── backend/                        # FastAPI
    │   ├── main.py                     # Point d'entrée (port 8000)
    │   ├── database.py                 # SQLAlchemy setup
    │   ├── api/
    │   │   ├── routes/
    │   │   │   ├── auth.py             # Auth + assignation QCM
    │   │   │   ├── mcq.py              # Récupération QCM
    │   │   │   ├── validations.py      # Stockage validations
    │   │   │   ├── generation.py       # Génération custom
    │   │   │   ├── admin.py            # Dashboard admin
    │   │   │   └── answerability.py    # Tests answerability
    │   │   ├── models/
    │   │   │   ├── auth.py             # Pydantic models (User, Token...)
    │   │   │   └── db_models.py        # SQLAlchemy (MCQAssignment, Validation)
    │   │   └── utils/
    │   │       ├── dependencies.py     # JWT auth dependency
    │   │       ├── security.py         # Hashing, token creation
    │   │       └── generate_mcq_GPU.py # Client Ollama
    │   └── eval/
    │       ├── eval_dataframe.py       # Orchestrateur du pipeline
    │       ├── ambiguity.py            # Métrique B2
    │       ├── distractors_quality.py  # Métrique B3
    │       ├── relevance.py            # Métrique B1
    │       ├── originality.py          # Métrique A3
    │       ├── readability.py          # Métrique A4
    │       ├── disclosure.py           # Métrique B4
    │       ├── difficulty.py           # Métrique B5
    │       ├── answerability.py        # Métrique B6
    │       ├── negation.py             # Métrique A2
    │       ├── question_check.py       # Métrique A1
    │       └── utils.py                # Chargement modèles, embeddings
    │
    └── frontend/mcq-evalutation/src/app/
        ├── pages/
        │   ├── login-page/
        │   ├── dashboard-page/         # Hub principal
        │   ├── evaluation-page/        # Interface d'évaluation
        │   ├── history-page/           # Historique des validations
        │   ├── admin-dashboard-page/   # Gestion admin
        │   └── answerability-page/     # Tests answerability
        ├── components/
        │   ├── mcq-selection-modal/    # Assignation de QCM
        │   └── generate-from-material-modal/  # Génération custom
        ├── services/
        │   ├── auth.service.ts
        │   ├── mcq.service.ts
        │   ├── validation.service.ts
        │   └── admin.service.ts
        └── guards/
            ├── auth.guard.ts
            └── admin.guard.ts
```

---

## API Reference

| Méthode | Endpoint | Description | Auth |
|---------|----------|-------------|------|
| POST | `/api/auth/login` | Authentification JWT | — |
| GET | `/api/auth/me` | Utilisateur courant | ✅ |
| PUT | `/api/auth/change-password` | Changer le mot de passe | ✅ |
| POST | `/api/auth/assign-mcq` | Assigner des QCM à l'utilisateur | ✅ |
| GET | `/api/mcq/assigned` | QCM assignés à l'utilisateur | ✅ |
| GET | `/api/mcq/{id}?model=X` | Récupérer un QCM avec ses métriques | ✅ |
| GET | `/api/mcq-models` | Modèles disponibles + compteurs | ✅ |
| POST | `/api/validations/{id}/validate` | Soumettre une validation | ✅ |
| GET | `/api/validations/history` | Historique des validations | ✅ |
| GET | `/api/validations/stats` | Statistiques utilisateur | ✅ |
| POST | `/api/generate` | Démarrer une génération custom | ✅ |
| GET | `/api/generate/{job_id}` | Statut de la génération | ✅ |
| POST | `/api/answerability/start` | Lancer un test answerability | ✅ |
| GET | `/api/admin/stats/global` | Stats globales | ✅ 👑 |
| GET | `/api/admin/stats/by-model` | Stats par modèle | ✅ 👑 |
| GET | `/api/admin/users` | Gestion utilisateurs | ✅ 👑 |
| GET | `/api/admin/export` | Export CSV des validations | ✅ 👑 |
| GET/PUT | `/api/admin/tracker` | Suivi des assignations | ✅ 👑 |

*👑 = rôle admin requis — Documentation interactive : `http://localhost:8000/docs`*

---

## Frontend

### Pages

| Route | Accès | Description |
|-------|-------|-------------|
| `/login` | Public | Connexion |
| `/dashboard` | Évaluateur | Stats perso, récupération et génération de QCM |
| `/evaluation` | Évaluateur | Interface de validation (Sections A & B) |
| `/history` | Évaluateur | Historique des validations avec filtre ACCEPT/REJECT |
| `/admin` | Admin | Statistiques, gestion utilisateurs, export CSV |
| `/answerability` | Admin | Configuration des tests answerability |

### Interface d'évaluation

L'interface de validation est structurée en deux sections :

- **Section A** (Format) : 4 checks automatiques avec possibilité de confirmer/rejeter chaque check
- **Section B** (Contenu) : 6 checks avec scores et seuils visibles
- **Décision finale** : ACCEPT / REJECT avec feedback libre
- **Mode révision** : Accessible via `?mcq_id=XXX&model=YYY` pour réévaluer un QCM existant

---

## Génération de QCM personnalisés

Les évaluateurs peuvent soumettre leur propre contenu textuel pour générer des QCM :

1. Sur le dashboard, cliquer sur **"Générer depuis un document"**
2. Coller le texte source (ou uploader un PDF)
3. Le système génère 1 QCM via Ollama en arrière-plan
4. Le QCM est automatiquement assigné à l'utilisateur et évalué
5. Identifiants au format `CSTM-XXXXXXX-1`, stockés dans `data/custom_mcqs/`

---

## Pipeline de fine-tuning

Les validations humaines sont stockées avec le contenu source (`content_raw`) et la décision (`ACCEPT`/`REJECT`), permettant deux types de fine-tuning :

- **SFT** (Supervised Fine-Tuning) : Entraîner sur les QCM acceptés uniquement
- **DPO** (Direct Preference Optimization) : Utiliser les paires (accepté, rejeté) pour le même contexte source comme signal de préférence

Voir `notebooks/dpo_training.ipynb` pour l'implémentation.

---

## Configuration

### Variables d'environnement (`src/page/backend/.env`)

```env
# Sécurité
SECRET_KEY=<clé_aléatoire_longue>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=480

# Base de données
DATABASE_URL=sqlite:///./mcq_evaluation.db

# APIs externes
OPENAI_API_KEY=sk-...            # Pour les métriques GPT-4o (B3, B4, B5)
HF_TOKEN=hf_...                  # Pour MedMCQA (answerability)

# Serveurs LLM
OLLAMA_HOST=http://localhost:11434
VLLM_HOST=http://localhost:8001

# CORS
ALLOWED_ORIGINS=http://localhost:4200

# Environnement
ENVIRONMENT=development
```

### Gestion des utilisateurs

```bash
cd src/page/backend
python generate_passwords.py
```

Structure de `data/users.json` :

```json
[
  {
    "username": "evaluateur1",
    "email": "eval1@example.com",
    "role": "evaluator",
    "password": "mot_de_passe"
  },
  {
    "username": "admin",
    "email": "admin@example.com",
    "role": "admin",
    "password": "mot_de_passe_admin"
  }
]
```

---

## Sécurité

- Mots de passe hashés via **bcrypt** (passlib)
- Tokens JWT HS256 avec expiration (8h par défaut)
- Rate limiting sur `/api/auth/login` (5 req/min)
- Headers de sécurité HTTP (HSTS, X-Frame-Options, CSP)
- Routes admin protégées par guard `role=admin`
- Logout côté client uniquement (architecture stateless)
