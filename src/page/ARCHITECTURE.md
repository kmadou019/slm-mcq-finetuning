# Architecture MCQ Evaluation App

## 🎯 Vue d'ensemble

Application complète pour valider des questions à choix multiples (MCQ) générées automatiquement, basée sur des objectifs de connaissance (LISA Sheets).

**Fonctionnalités principales :**
- 🔐 Authentification des évaluateurs
- 📊 Dashboard avec statistiques
- ✅ Validation interactive des MCQ
- 📝 Feedback et commentaires
- 📥 Export des résultats
- 🎯 Sélection du nombre de questions à évaluer

## 📐 Architecture Technique

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Angular 21)                     │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │   Components   │  │    Services    │  │     Models     │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/REST
┌──────────────────────────┴──────────────────────────────────┐
│                   Backend (FastAPI/Python)                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │   Endpoints    │  │  Business Logic │  │    Database    │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎨 Frontend Angular

### 0. Flux utilisateur 🆕

```
1. Évaluateur arrive sur /login
2. Se connecte avec identifiants
3. Modal: Sélectionne le nombre de MCQ à évaluer (ex: 10, 20, 50)
4. Redirigé vers /dashboard avec ses MCQ assignées
5. Dashboard affiche les statistiques et la liste des MCQ
6. Clique sur "Commencer l'évaluation" → /evaluation
7. Valide les MCQ une par une avec navigation
8. Export des résultats depuis le dashboard
9. Déconnexion
```

### 1. Models/Interfaces TypeScript

#### `models/auth.model.ts` 🆕
```typescript
export interface User {
  id: string;
  username: string;
  email: string;
  role: 'evaluator' | 'admin';
  created_at: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface MCQAssignment {
  user_id: string;
  mcq_count: number;
  assigned_mcq_ids: string[];
  assigned_at: string;
  status: 'pending' | 'in_progress' | 'completed';
}

export interface MCQSelectionRequest {
  count: number;
}
```

#### `models/mcq.model.ts`
```typescript
export interface MCQOption {
  A: string;
  B: string;
  C: string;
  D: string;
}

export interface SectionCheck {
  dimension: string;
  result: 'PASS' | 'WARN';
  threshold: string;
  notes: string;
  validated?: boolean;
}

export interface MCQCard {
  item_id: string;
  source_material: string;
  generator_info: string;
  output_format: string;
  mcq_question: string;
  options: MCQOption;
  correct_option: string;
  section_a_checks: SectionCheck[];
  section_b_checks: SectionCheck[];
  decision_policy: string;
  final_decision: 'ACCEPT' | 'REVISE';
  audit_trail: string;
  lisa_texte_brut: string;
}

export interface LISASheet {
  identifiant: string;
  rang: string;
  intitule: string;
  description: string;
  rubrique: string;
  item_parent: string;
  contenu: string;
}

export interface ValidationData {
  index: number;
  human_decision: 'ACCEPT' | 'REJECT' | null;
  human_feedback: string;
  validated_fields: Record<string, boolean>;
  timestamp: string;
}
```

#### `models/api.model.ts`
```typescript
export interface MCQListResponse {
  total: number;
  cards: MCQCard[];
}

export interface ValidationSubmission {
  mcq_id: string;
  decision: 'ACCEPT' | 'REJECT';
  feedback: string;
  validated_checks: Record<string, boolean>;
}

export interface ExportData {
  validations: ValidationSubmission[];
  export_date: string;
}
```

### 2. Composants Angular

#### Structure des composants
```
src/app/
├── components/
│   ├── mcq-card/                      # Carte MCQ
│   │   ├── mcq-card.component.ts
│   │   ├── mcq-card.component.html
│   │   └── mcq-card.component.scss
│   ├── lisa-sheet/                    # LISA Sheet
│   │   ├── lisa-sheet.component.ts
│   │   ├── lisa-sheet.component.html
│   │   └── lisa-sheet.component.scss
│   ├── mcq-list/                      # Liste MCQ
│   │   ├── mcq-list.component.ts
│   │   ├── mcq-list.component.html
│   │   └── mcq-list.component.scss
│   ├── navigation/                    # Navigation slides
│   │   ├── navigation.component.ts
│   │   ├── navigation.component.html
│   │   └── navigation.component.scss
│   ├── validation-form/               # Formulaire validation
│   │   ├── validation-form.component.ts
│   │   ├── validation-form.component.html
│   │   └── validation-form.component.scss
│   ├── mcq-selection-modal/           # 🆕 Modal sélection nombre MCQ
│   │   ├── mcq-selection-modal.component.ts
│   │   ├── mcq-selection-modal.component.html
│   │   └── mcq-selection-modal.component.scss
│   ├── header/                        # 🆕 Header avec user info
│   │   ├── header.component.ts
│   │   ├── header.component.html
│   │   └── header.component.scss
│   └── stats-card/                    # 🆕 Carte statistiques
│       ├── stats-card.component.ts
│       ├── stats-card.component.html
│       └── stats-card.component.scss
├── pages/
│   ├── login-page/                    # 🆕 Page de connexion
│   │   ├── login-page.component.ts
│   │   ├── login-page.component.html
│   │   └── login-page.component.scss
│   ├── dashboard-page/                # Dashboard principal
│   │   ├── dashboard-page.component.ts
│   │   ├── dashboard-page.component.html
│   │   └── dashboard-page.component.scss
│   └── evaluation-page/               # Page évaluation
│       ├── evaluation-page.component.ts
│       ├── evaluation-page.component.html
│       └── evaluation-page.component.scss
├── guards/                            # 🆕 Guards de route
│   └── auth.guard.ts
├── interceptors/                      # 🆕 HTTP Interceptors
│   └── auth.interceptor.ts
├── services/
│   ├── auth.service.ts                # 🆕 Service authentification
│   ├── mcq.service.ts                 # Service MCQ
│   ├── validation.service.ts          # Service validation
│   └── storage.service.ts             # Service stockage local
└── models/
    ├── auth.model.ts                  # 🆕 Models authentification
    ├── mcq.model.ts                   # Models MCQ
    └── api.model.ts                   # Models API
```

#### Composants détaillés

**1. MCQCardComponent** (`mcq-card`)
- **Rôle**: Affiche une carte MCQ avec tous ses détails
- **Inputs**:
  - `card: MCQCard`
  - `index: number`
- **Outputs**:
  - `acceptCard: EventEmitter<number>`
  - `rejectCard: EventEmitter<number>`
  - `feedbackChange: EventEmitter<{index: number, feedback: string}>`
  - `checkboxChange: EventEmitter<{index: number, check: string, value: boolean}>`
- **Fonctionnalités**:
  - Affichage de la question et options
  - Tableaux Section A et B avec checkboxes
  - Boutons Accept/Reject
  - Zone de feedback (textarea)
  - Styling conditionnel (PASS/WARN)

**2. LISASheetComponent** (`lisa-sheet`)
- **Rôle**: Affiche une LISA Sheet parsée
- **Inputs**:
  - `lisaData: LISASheet`
- **Fonctionnalités**:
  - Affichage structuré des métadonnées
  - Affichage du contenu formaté
  - Styling dédié LISA

**3. MCQListComponent** (`mcq-list`)
- **Rôle**: Liste/aperçu de toutes les MCQ
- **Inputs**:
  - `cards: MCQCard[]`
- **Outputs**:
  - `selectCard: EventEmitter<number>`
- **Fonctionnalités**:
  - Liste scrollable des MCQ
  - Indicateurs de statut (validated, accepted, rejected)
  - Navigation vers une carte spécifique

**4. NavigationComponent** (`navigation`)
- **Rôle**: Contrôles de navigation entre les cartes
- **Inputs**:
  - `currentIndex: number`
  - `totalCards: number`
- **Outputs**:
  - `next: EventEmitter<void>`
  - `previous: EventEmitter<void>`
  - `goTo: EventEmitter<number>`
  - `export: EventEmitter<void>`
- **Fonctionnalités**:
  - Boutons Précédent/Suivant
  - Input pour aller à une carte spécifique
  - Barre de progression
  - Bouton Export
  - Contrôles clavier (flèches)

**5. ValidationFormComponent** (`validation-form`)
- **Rôle**: Formulaire de validation pour une carte
- **Inputs**:
  - `cardIndex: number`
- **Outputs**:
  - `submit: EventEmitter<ValidationSubmission>`
- **Fonctionnalités**:
  - Checkboxes pour Section A et B
  - Boutons Accept/Reject
  - Textarea feedback
  - Validation du formulaire

**6. EvaluationPageComponent** (`evaluation-page`)
- **Rôle**: Page principale d'évaluation (conteneur)
- **Fonctionnalités**:
  - Gestion de l'état global
  - Coordination des composants enfants
  - Système de slides/pagination
  - Sauvegarde automatique

**7. DashboardPageComponent** (`dashboard-page`)
- **Rôle**: Tableau de bord avec statistiques
- **Fonctionnalités**:
  - Statistiques globales (acceptées/refusées/en attente)
  - Liste des MCQ assignées à l'évaluateur
  - Filtres (status, date)
  - Bouton "Commencer l'évaluation"
  - Progression de l'évaluation

**8. LoginPageComponent** (`login-page`) 🆕
- **Rôle**: Page de connexion des évaluateurs
- **Fonctionnalités**:
  - Formulaire login (username + password)
  - Validation des champs
  - Gestion des erreurs (credentials invalides)
  - Redirection vers modal de sélection après login réussi
  - Remember me (optionnel)

**9. MCQSelectionModalComponent** (`mcq-selection-modal`) 🆕
- **Rôle**: Modal pour sélectionner le nombre de MCQ à évaluer
- **Inputs**:
  - `isOpen: boolean`
- **Outputs**:
  - `select: EventEmitter<number>`
  - `close: EventEmitter<void>`
- **Fonctionnalités**:
  - Options prédéfinies (10, 20, 50, 100)
  - Input personnalisé
  - Validation du nombre
  - Affichage du nombre de MCQ disponibles
  - Bouton "Commencer"

**10. HeaderComponent** (`header`) 🆕
- **Rôle**: Header de l'application avec informations utilisateur
- **Fonctionnalités**:
  - Logo/Titre de l'application
  - Nom de l'évaluateur connecté
  - Menu dropdown (profil, déconnexion)
  - Navigation (Dashboard, Évaluation)

**11. StatsCardComponent** (`stats-card`) 🆕
- **Rôle**: Carte de statistiques réutilisable
- **Inputs**:
  - `title: string`
  - `value: number`
  - `icon: string`
  - `color: string`
- **Fonctionnalités**:
  - Affichage stylisé d'une métrique
  - Animation au hover

### 3. Services Angular

#### `services/auth.service.ts` 🆕
```typescript
@Injectable({ providedIn: 'root' })
export class AuthService {
  private apiUrl = environment.apiUrl;
  private currentUserSubject = new BehaviorSubject<User | null>(null);
  public currentUser$ = this.currentUserSubject.asObservable();

  // POST /api/auth/login - Connexion
  login(credentials: LoginRequest): Observable<LoginResponse>

  // POST /api/auth/logout - Déconnexion
  logout(): Observable<void>

  // GET /api/auth/me - Utilisateur actuel
  getCurrentUser(): Observable<User>

  // POST /api/auth/assign-mcq - Assigner des MCQ
  assignMCQs(count: number): Observable<MCQAssignment>

  // Vérifier si l'utilisateur est connecté
  isAuthenticated(): boolean

  // Récupérer le token
  getToken(): string | null

  // Sauvegarder le token
  saveToken(token: string): void

  // Supprimer le token
  removeToken(): void
}
```

#### `services/mcq.service.ts`
```typescript
@Injectable({ providedIn: 'root' })
export class McqService {
  private apiUrl = environment.apiUrl;

  // GET /api/mcq - Liste toutes les MCQ
  getMCQList(): Observable<MCQListResponse>

  // GET /api/mcq/:id - Détails d'une MCQ
  getMCQById(id: string): Observable<MCQCard>

  // POST /api/mcq - Créer une nouvelle MCQ
  createMCQ(data: CreateMCQDto): Observable<MCQCard>

  // DELETE /api/mcq/:id - Supprimer une MCQ
  deleteMCQ(id: string): Observable<void>
}
```

#### `services/validation.service.ts`
```typescript
@Injectable({ providedIn: 'root' })
export class ValidationService {
  private apiUrl = environment.apiUrl;

  // POST /api/validation - Soumettre une validation
  submitValidation(data: ValidationSubmission): Observable<void>

  // GET /api/validation - Liste des validations
  getValidations(): Observable<ValidationData[]>

  // POST /api/validation/export - Exporter les validations
  exportValidations(): Observable<Blob>
}
```

#### `services/storage.service.ts`
```typescript
@Injectable({ providedIn: 'root' })
export class StorageService {
  // Gestion du localStorage pour persistance locale
  saveValidation(index: number, data: ValidationData): void
  loadValidations(): Record<number, ValidationData>
  clearValidations(): void

  // Auto-save avec debounce
  autoSave(data: any): void
}
```

### 4. Guards & Interceptors 🆕

#### `guards/auth.guard.ts`
```typescript
@Injectable({ providedIn: 'root' })
export class AuthGuard implements CanActivate {
  constructor(
    private authService: AuthService,
    private router: Router
  ) {}

  canActivate(route: ActivatedRouteSnapshot, state: RouterStateSnapshot): boolean {
    if (this.authService.isAuthenticated()) {
      return true;
    }

    // Redirection vers login avec returnUrl
    this.router.navigate(['/login'], {
      queryParams: { returnUrl: state.url }
    });
    return false;
  }
}
```

#### `interceptors/auth.interceptor.ts`
```typescript
@Injectable()
export class AuthInterceptor implements HttpInterceptor {
  constructor(private authService: AuthService) {}

  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    const token = this.authService.getToken();

    if (token) {
      req = req.clone({
        setHeaders: {
          Authorization: `Bearer ${token}`
        }
      });
    }

    return next.handle(req).pipe(
      catchError((error: HttpErrorResponse) => {
        if (error.status === 401) {
          // Token expiré ou invalide
          this.authService.logout();
          this.router.navigate(['/login']);
        }
        return throwError(() => error);
      })
    );
  }
}
```

### 5. Routing

#### `app.routes.ts` 🔄
```typescript
export const routes: Routes = [
  { path: '', redirectTo: '/login', pathMatch: 'full' },
  {
    path: 'login',
    component: LoginPageComponent
  },
  {
    path: 'dashboard',
    component: DashboardPageComponent,
    canActivate: [AuthGuard]
  },
  {
    path: 'evaluation',
    component: EvaluationPageComponent,
    canActivate: [AuthGuard]
  },
  {
    path: 'evaluation/:id',
    component: EvaluationPageComponent,
    canActivate: [AuthGuard]
  },
  { path: '**', redirectTo: '/login' }
];
```

---

## 🔧 Backend FastAPI

### 1. Structure du Backend

```
backend/
├── main.py                 # Point d'entrée FastAPI
├── card.py                 # Code existant (parsing, génération)
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── mcq.py          # Routes MCQ
│   │   └── validation.py   # Routes validation
│   ├── models/
│   │   ├── mcq.py          # Modèles Pydantic
│   │   └── validation.py   # Modèles validation
│   └── services/
│       ├── mcq_service.py  # Business logic MCQ
│       └── validation_service.py
├── database/
│   ├── __init__.py
│   ├── db.py               # Configuration DB
│   └── models.py           # Modèles SQLAlchemy
└── requirements.txt
```

### 2. API Endpoints

#### MCQ Endpoints

**GET /api/mcq**
- Description: Liste toutes les MCQ
- Query params:
  - `skip: int = 0`
  - `limit: int = 100`
  - `status: Optional[str]` (ACCEPT, REVISE, PENDING)
- Response: `MCQListResponse`

**GET /api/mcq/{mcq_id}**
- Description: Détails d'une MCQ spécifique
- Response: `MCQCard`

**POST /api/mcq**
- Description: Créer une nouvelle MCQ
- Body: `CreateMCQDto`
- Response: `MCQCard`

**DELETE /api/mcq/{mcq_id}**
- Description: Supprimer une MCQ
- Response: `204 No Content`

**POST /api/mcq/generate**
- Description: Générer des MCQ depuis un CSV
- Body: `file: UploadFile`, `model: str`
- Response: `{ count: int, cards: MCQCard[] }`

#### Validation Endpoints

**POST /api/validation**
- Description: Soumettre une validation
- Body: `ValidationSubmission`
- Response: `201 Created`

**GET /api/validation**
- Description: Liste des validations
- Query params:
  - `mcq_id: Optional[str]`
  - `decision: Optional[str]`
- Response: `ValidationData[]`

**POST /api/validation/export**
- Description: Exporter les validations en JSON
- Query params:
  - `format: str = 'json'` (json, csv)
- Response: `FileResponse`

**GET /api/validation/stats**
- Description: Statistiques des validations
- Response:
```json
{
  "total": 100,
  "accepted": 75,
  "rejected": 20,
  "pending": 5
}
```

#### LISA Sheet Endpoints

**GET /api/lisa/{id}**
- Description: Récupérer une LISA Sheet parsée
- Response: `LISASheet`

**POST /api/lisa/parse**
- Description: Parser un texte LISA Sheet
- Body: `{ text: str }`
- Response: `LISASheet`

### 3. Modèles Pydantic

#### `api/models/mcq.py`
```python
from pydantic import BaseModel
from typing import Dict, List, Optional, Literal

class MCQOption(BaseModel):
    A: str
    B: str
    C: str
    D: str

class SectionCheck(BaseModel):
    dimension: str
    result: Literal['PASS', 'WARN']
    threshold: str
    notes: str

class MCQCard(BaseModel):
    item_id: str
    source_material: str
    generator_info: str
    output_format: str
    mcq_question: str
    options: MCQOption
    correct_option: str
    section_a_checks: List[SectionCheck]
    section_b_checks: List[SectionCheck]
    decision_policy: str
    final_decision: Literal['ACCEPT', 'REVISE']
    audit_trail: str
    lisa_texte_brut: str

class CreateMCQDto(BaseModel):
    question: str
    options: MCQOption
    correct_option: str
    source_material: str
    generator_info: str
```

#### `api/models/validation.py`
```python
from pydantic import BaseModel
from typing import Dict, Optional, Literal
from datetime import datetime

class ValidationSubmission(BaseModel):
    mcq_id: str
    decision: Literal['ACCEPT', 'REJECT']
    feedback: Optional[str] = None
    validated_checks: Dict[str, bool]

class ValidationData(BaseModel):
    id: int
    mcq_id: str
    decision: Literal['ACCEPT', 'REJECT']
    feedback: Optional[str]
    validated_checks: Dict[str, bool]
    timestamp: datetime
    user_id: Optional[str] = None
```

### 4. Base de données

#### Schéma (SQLite/PostgreSQL)

**Table: mcq_cards**
```sql
CREATE TABLE mcq_cards (
    id VARCHAR(50) PRIMARY KEY,
    source_material VARCHAR(100),
    generator_info VARCHAR(100),
    question TEXT NOT NULL,
    option_a TEXT,
    option_b TEXT,
    option_c TEXT,
    option_d TEXT,
    correct_option VARCHAR(1),
    decision_policy TEXT,
    final_decision VARCHAR(10),
    audit_trail TEXT,
    lisa_raw_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Table: validations**
```sql
CREATE TABLE validations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mcq_id VARCHAR(50),
    decision VARCHAR(10) NOT NULL,
    feedback TEXT,
    validated_checks JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (mcq_id) REFERENCES mcq_cards(id)
);
```

**Table: section_checks**
```sql
CREATE TABLE section_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mcq_id VARCHAR(50),
    section VARCHAR(1), -- 'A' or 'B'
    dimension VARCHAR(100),
    result VARCHAR(10),
    threshold VARCHAR(50),
    notes TEXT,
    FOREIGN KEY (mcq_id) REFERENCES mcq_cards(id)
);
```

---

## 🔄 Flux de données

### 1. Chargement initial
```
User → Frontend → GET /api/mcq → Backend → Database → Response → Frontend → Display
```

### 2. Validation d'une carte
```
User interacts → Frontend (validation form) → POST /api/validation → Backend → Save to DB → Response → Update UI
```

### 3. Export des validations
```
User clicks Export → Frontend → POST /api/validation/export → Backend → Generate JSON → Download file
```

### 4. Génération de nouvelles MCQ
```
Upload CSV → POST /api/mcq/generate → Backend (card.py logic) → Save to DB → Return cards
```

---

## 🎯 Phases d'implémentation

### Phase 1 : Setup et Models ✅ (Fait)
- [x] Créer structure backend/frontend
- [x] Installer Angular CLI et dépendances
- [ ] Définir models TypeScript
- [ ] Définir models Pydantic

### Phase 2 : Backend API
- [ ] Setup FastAPI main.py
- [ ] Créer endpoints MCQ
- [ ] Créer endpoints validation
- [ ] Setup base de données SQLite
- [ ] Intégrer code card.py existant
- [ ] Tests API avec Swagger UI

### Phase 3 : Frontend Core
- [ ] Créer composants de base
- [ ] Créer services Angular
- [ ] Implémenter routing
- [ ] Setup HttpClient et intercepteurs

### Phase 4 : Features principales
- [ ] MCQ Card avec validation
- [ ] LISA Sheet display
- [ ] Navigation entre cartes
- [ ] Sauvegarde localStorage
- [ ] Export JSON

### Phase 5 : Polish et Features avancées
- [ ] Dashboard avec stats
- [ ] Filtres et recherche
- [ ] Responsive design
- [ ] Animations et transitions
- [ ] Tests unitaires

---

## 🛠️ Stack Technique

### Frontend
- **Framework**: Angular 21
- **Language**: TypeScript 5
- **Styling**: SCSS
- **HTTP**: HttpClient (Angular)
- **State**: Services + RxJS
- **Storage**: LocalStorage API

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.10+
- **Database**: SQLite (dev) / PostgreSQL (prod)
- **ORM**: SQLAlchemy
- **Validation**: Pydantic
- **CORS**: fastapi-cors

### DevOps
- **Package Manager**: pnpm (frontend)
- **Python Env**: venv / conda
- **API Docs**: Swagger UI (auto-generated)
- **Git**: Version control

---

## 📝 Prochaines étapes

**Que voulez-vous faire en premier ?**

1. **Créer les models TypeScript** dans Angular
2. **Setup le backend FastAPI** avec les premiers endpoints
3. **Créer le premier composant** (MCQCard ou LISASheet)
4. **Autre chose ?**

