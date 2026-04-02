---
name: mcq-prompt-builder
description: Generate a prompt to create a single French ECNi QCM question (QRU, 4 propositions).
allowed-tools: Read, Write, mcp__mcp-graphdb__sparqlQuery, mcp__mcp-graphdb__listGraphs
---

# MCQ Prompt Builder (French ECNi — Question Isolée)

Generate a detailed prompt for AI to create a single French medical exam question (QCM format).

## Purpose

This system prompt builds a prompt that can be given to any AI to generate one medical exam question. It does NOT generate the question itself - it creates the instruction prompt.

## Usage Workflow

1. **Upload medical content** — extract full text from PDF/DOCX
2. **Get reference and objectives** — GraphDB search or manual entry; display for user validation
3. **Ask configuration** — objective, specific instructions
4. **Build and output prompt** — ready to copy/paste to AI

---

## EMBEDDED CONFIGURATION

All rules are embedded - do NOT ask user to provide them.

### Exam Format: QI (Question Isolée)

Authorized question type: QRU only

- **1 question** generated per prompt
- The question is standalone — no clinical case narrative required

### Question Format

- **QRU**: 4 propositions, 1 exacte

---

## GRAPHDB CONNECTION

Use the `mcp__mcp-graphdb__sparqlQuery` MCP tool directly. No session initialization or Bearer token management is required — the MCP server handles authentication transparently.

To execute a SPARQL query, call:

```
mcp__mcp-graphdb__sparqlQuery(query: "{SPARQL_QUERY}", format: "json")
```

To list available graphs, call:

```
mcp__mcp-graphdb__listGraphs()
```

---

## GRAPHDB SPARQL STRATEGY

The LISA GraphDB uses a 3-tier structure:

```
KnowledgeItem (e.g., "Facteurs de risque cardio-vasculaire")
    ↓ lisa:hasKnowledgeObjective
KnowledgeObjective (e.g., "Connaître les facteurs de risque")
    ↓ attributes
label, rank [A/B], description, order, comment, fullWikitext
```

All queries are executed via `mcp__mcp-graphdb__sparqlQuery`. Call it directly for each query — no session management needed.

### Query 1 — Search KnowledgeItems (Hop 0)

```sparql
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX lisa: <https://uness.fr/lisa/ontology/>

SELECT ?item ?label WHERE {
  ?item rdfs:label ?label .
  ?item rdf:type lisa:KnowledgeItem .
  FILTER(regex(?label, "{search_term}", "i"))
} LIMIT 10
```

Returns: list of `?item` (URI) and `?label`. Extract item ID from URI suffix (e.g. `KnowledgeItem222` → "Item 222").

### Query 2 — Get Objective URIs for an Item (Hop 1)

```sparql
PREFIX lisa: <https://uness.fr/lisa/ontology/>

SELECT ?objective WHERE {
  <{item_uri}> lisa:hasKnowledgeObjective ?objective .
}
```

Returns: list of `?objective` URIs.

### Query 3 — Get Attributes for Each Objective (Hop 2)

Run once per objective URI from Query 2:

```sparql
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX lisa: <https://uness.fr/lisa/ontology/>

SELECT ?property ?value WHERE {
  <{objective_uri}> ?property ?value .
}
```

Map properties to fields:

| Property suffix              | Field                     |
|------------------------------+---------------------------|
| `rdfs#label`                 | label                     |
| `lisa/ontology/rank`         | rank (A or B)             |
| `lisa/ontology/description`  | description               |
| `lisa/ontology/order`        | order (integer, sort key) |
| `rdfs#comment`               | comment                   |
| `lisa/ontology/fullWikitext` | fullWikitext              |

Sort objectives by `order` ascending. Format each as: `{label} [{rank}]`

### Search Fallback Strategy

If Query 1 returns empty results, retry with variations before giving up:
1. Truncate term: `"immunologie"` → `"immun"`, `"rétinopathie"` → `"rétin"`
2. Remove accents: `"réaction"` → `"reaction"`
3. Try a synonym or related term
4. Try the ECNi item number directly (e.g. `"222"`)

### Best Practices

- Always verify selected item with user before running Hop 1+2
- [A] = essential, [B] = recommended
- Don't re-query the same item in the same session

---

## PROMPT BUILDING WORKFLOW

### Step 1: Upload Medical Content

Ask for a PDF/DOCX file and extract full text content.

Use the `document-skills:pdf` Claude skill to extract text from PDF files.

This content will be used in the final prompt section:
```
### CONTENU INTÉGRAL DE LA SOURCE
— [Item reference] —
[FULL EXTRACTED TEXT]
```

### Step 2: Get Reference and Objectives

**Ask user**: "Do you want to search for the official knowledge item?"

If **YES** (GraphDB):
1. Ask for search term (e.g., "vasculaire", "item 222", "immunologie")
2. Execute Query 1 (Hop 0) with the term — apply Search Fallback Strategy if empty
3. Display results for user selection:
   ```
   1. [KnowledgeItem120] Réaction inflammatoire : aspects biologiques et cliniques
   2. [KnowledgeItem134] Syndrome inflammatoire biologique
   ```
4. User selects one item
5. Execute Query 2 (Hop 1) to get objective URIs, then Query 3 (Hop 2) for each URI — sort by `order`, format as `{label} [{rank}]`
6. Extract reference from URI (KnowledgeItem120 → "Item 120")

If **NO** (Manual entry):
1. Ask user to enter reference manually (e.g., "Semio_Arterielle")
2. Parse objectives from PDF content extracted in Step 1
3. Look for patterns: `• Objective [A]`, `# Objective [B]`, `1. Objective [A]`
4. If no objectives can be extracted, ask user to enter them manually

### Step 3: Display and Validate Objectives

Show objectives list with ranks to user. User can edit/confirm.

### Step 4: Ask Question Configuration

Ask:
1. **Which objective to use** — user selects from the validated list (Rang A or B)
2. **Format** — QRU (fixe, ne pas demander à l'utilisateur)
3. **Specific instructions** (optional)

**Rang definition:**
- **Rang A** — Essentiel : objectifs fondamentaux que tout étudiant doit maîtriser
- **Rang B** — Approfondi : objectifs complémentaires pour une maîtrise avancée

### Step 5: Build Prompt

```
### CONSIGNES DE RÉDACTION

Consignes pour la rédaction de la question :
- Respecter exclusivement les informations et objectifs explicitement présents dans l'item fourni
- Rédiger en français clair, précis et sans ambiguïté
- Écarter toute information hors du périmètre de l'item et de l'objectif choisis

Consignes pour la rédaction des propositions :
- Les propositions doivent être homogènes, parallèles et d'un niveau de granularité similaire
- Les propositions doivent être exprimées à la forme affirmative
- Les propositions ne doivent pas apporter d'information complémentaire
- Les propositions incorrectes (distracteurs) doivent être plausibles mais fausses
- Les propositions doivent être courtes et concises
- Fournir une justification pédagogique pour chaque proposition (pourquoi elle est correcte ou incorrecte)
- Fournir un commentaire global pour la question (ce qu'elle évalue ou un piège courant)

### FORMAT

- Type: QI (Question Isolée)
- Format: QRU

### ITEM ET DOCUMENT

- [Item reference] — [filename]

### OBJECTIFS (liste de l'item)

[Item reference]:
- [Objective 1 [A]]
- [Objective 2 [B]]
[...]

### OBJECTIF CIBLÉ

Ref: [reference] | Obj: [selected objective] | Rang: [A/B] | Format: QRU

### CONSIGNES SPÉCIFIQUES

[User-provided specific instructions — omit section if none]

### FORMAT DE QUESTION

Format QRU : 4 propositions : une seule exacte

### SCHÉMA DE SORTIE ATTENDU

- T : titre de la question
- F : format (QRU)
- Ref : référence à l'item source
- O : objectif pédagogique
- Z : rang (A ou B)
- D : consignes spécifiques (si applicable)
- Q : texte de la question
- P01–P04 : 4 propositions (1 seule exacte)
- R : réponse correcte
- C : commentaire pédagogique

Respecter scrupuleusement ce format de sortie.

### CONTENU INTÉGRAL DE LA SOURCE

— [Item reference] —
[Full extracted text from PDF]
```

---

## COMPLETE EXAMPLE

**User request:**
```
Build a prompt for one QCM question from Semio_Arterielle.pdf
```

**Your workflow:**

1. Extract PDF content using `document-skills:pdf`
2. Search GraphDB or parse objectives manually; display for user validation
3. Ask which objective and format to use
4. Build and output prompt

**Expected output:**
```
### CONSIGNES DE RÉDACTION

Consignes pour la rédaction de la question :
- Respecter exclusivement les informations et objectifs explicitement présents dans l'item fourni
- Rédiger en français clair, précis et sans ambiguïté
- Écarter toute information hors du périmètre de l'item et de l'objectif choisis

Consignes pour la rédaction des propositions :
- Les propositions doivent être homogènes, parallèles et d'un niveau de granularité similaire
- Les propositions doivent être exprimées à la forme affirmative
- Les propositions ne doivent pas apporter d'information complémentaire
- Les propositions incorrectes (distracteurs) doivent être plausibles mais fausses
- Les propositions doivent être courtes et concises
- Fournir une justification pédagogique pour chaque proposition (pourquoi elle est correcte ou incorrecte)
- Fournir un commentaire global pour la question (ce qu'elle évalue ou un piège courant)

### FORMAT

- Type: QI (Question Isolée)
- Format: QRU

### ITEM ET DOCUMENT

- Semio_Arterielle — Semio_Arterielle.pdf

### OBJECTIFS (liste de l'item)

Semio_Arterielle:
- Mener un interrogatoire vasculaire complet [A]
- Réaliser un examen physique vasculaire systématique [A]
- Reconnaître les principaux signes des pathologies vasculaires [A]
- Identifier les facteurs de risque cardiovasculaires [B]

### OBJECTIF CIBLÉ

Ref: Semio_Arterielle | Obj: Identifier les facteurs de risque cardiovasculaires [B] | Rang: B | Format: QRU

### CONSIGNES SPÉCIFIQUES

[omitted]

### FORMAT DE QUESTION

Format QRU : 4 propositions : une seule exacte

### SCHÉMA DE SORTIE ATTENDU

- T : titre de la question
- F : format (QRU)
- Ref : référence à l'item source
- O : objectif pédagogique
- Z : rang (A ou B)
- D : consignes spécifiques (si applicable)
- Q : texte de la question
- P01–P04 : 4 propositions (1 seule exacte)
- R : réponse correcte
- C : commentaire pédagogique

Respecter scrupuleusement ce format de sortie.

### CONTENU INTÉGRAL DE LA SOURCE

— Semio_Arterielle —
MANUEL DE SÉMIOLOGIE VASCULAIRE

OBJECTIFS D'APPRENTISSAGE
À la fin de ce manuel, vous devriez être capable de :
• Mener un interrogatoire vasculaire complet
• Réaliser un examen physique vasculaire systématique
[... full PDF content here ...]
```

---

## Output Options

After building the prompt, offer:
1. **Display prompt** (show in chat)
2. **Save to file** (save as .txt)
3. **Copy to clipboard** (if user wants to paste to external AI)

## Important Notes

- Validate format configuration before building
- Include full source content at end of prompt
