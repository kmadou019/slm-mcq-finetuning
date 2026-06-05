# Spike: Ollama Reasoning Models for MCQ Generation

**Date:** 2026-03-12
**Status:** In progress
**Goal:** Evaluate whether local reasoning models can improve MCQ generation quality by leveraging explicit chain-of-thought traces.

---

## Context

Current pipeline uses standard instruction-tuned models (e.g. `llama3.1:8b`, `medgemma-27b`) to generate MCQs from LISA sheets. Reasoning models expose an internal thinking trace (`message.thinking`) which could improve question coherence and distractor quality.

---

## What Was Explored

- Confirmed Ollama supports `"think": true` on the `/api/chat` endpoint
- Two reasoning models are available locally:
  - `magistral:latest` (24B, Mistral family) — compact, calculation-focused trace
  - `hf.co/unsloth/Qwen3-Next-80B-A3B-Thinking-GGUF:UD-Q4_K_XL` (80B, Qwen3) — verbose, metacognitive trace
- Both models return a separate `message.thinking` field distinct from `message.content`
- See [`runbook-ollama-reasoning-trace.md`](./runbook-ollama-reasoning-trace.md) for curl commands

---

## Evaluation Pipeline — Corrections Made

Following a scientific review of the distractor quality evaluation approach, the following issues were identified and fixed:

### Fixes implemented

| # | Issue | Fix |
|---|-------|-----|
| 1.1 | Position bias: generation model always puts correct answer in position `a`; GPT-4o over-scores first distractor | `shuffle_distractors()` added in `generate_mcq_GPU.py`, called after every `validate_mcq()` in generation pipeline and benchmark |
| 2.1 | Embedding model `BAAI/bge-base-en-v1.5` is English-only; corpus is French | Replaced with `almanach/moderncamembert-base` in `utils.py`, `ambiguity.py`, `relevance.py` |
| 2.3 | GPT-4o evaluated distractor quality without access to the LISA source content | `context_col='content_raw'` now passed to `generate_prompt_for_question()` in `distractors_quality.py` |
| 2.4 | Distractor quality binarised too early (single boolean), no per-distractor detail, fixed seuil regardless of LISA Rang | `distractors_quality.py` now stores raw scores + avg + justifications in `distractors_quality_detail` column. Threshold is rang-adaptive: `≥ 3` for Rang A, `≥ 4` for Rang B. Added rule `min(scores) ≥ 2` (no catastrophic distractor). |
| 3.1 | Answerability not shown in evaluation card | Added as **B5** in `section_b_checks` (both `mcq.py` and `generation.py`) |
| 3.2 | Ambiguity not shown in evaluation card | Added as **B6** in `section_b_checks` |
| 3.3 | B3 (Distractor plausibility) showed only `True/False` | Now shows `Rang X \| avg=X.XX \| scores=[x, x, x]` |

### Pending — ambiguity window threshold (Option A)

The ambiguity metric (B6) currently uses a single lower bound (`≥ 0.3 = PASS`). The full window `[0.3 – 0.75]` is not yet implemented. Justification from literature:

- **Mitkov, Ha & Karamanis (2009)** — intermediate similarity produces highest item discrimination
- **Yeung & Lee et al. (2019)** — explicit `STS < 80%` upper filter
- **Gierl et al. (2017)** — content similarity strategy: related but not synonymous
- **ArXiv 2025 (Student Choice Prediction)** — best distractors empirically cluster at intermediate cosine similarity

---

## Reliability Testing

The reliability of the GPT-4o distractor quality metric has not yet been validated empirically. Two tests are needed:

| Test | Method | Script |
|------|--------|--------|
| **Intra-rater** | Submit same MCQ twice to GPT-4o, measure score variance. Acceptable: mean abs diff < 0.5 | `notebooks/reliability_test.py` |
| **Inter-rater** | Annotate 30–50 MCQs manually, compute Cohen's κ vs GPT-4o. Target: κ ≥ 0.6 | `notebooks/reliability_test.py --kappa` |

---

## Benchmark Script

`notebooks/benchmark_nemotron_vs_magistral.py` — compares **Nemotron 30B** vs **Magistral 24B** on 10 LISA rows.

```bash
cd notebooks
python benchmark_nemotron_vs_magistral.py
# → docs/benchmark_nemotron_vs_magistral.html
```

**Metrics evaluated:** Originality, Readability (FK), Negation, Is-question, Relevance, Ambiguity (B6), Answerability (B5, GPT-4o), Disclosure (GPT-4o), Difficulty (GPT-4o), Distractor Quality with raw scores (GPT-4o).

**Output:** self-contained HTML — summary table (averages per metric) + per-question cards side by side.

---

## GPU Pipeline — Fixes et Améliorations (2026-04-10)

Corrections apportées lors du run de comparaison `compare_systems.sh` sur GPU (OAR job) :

| # | Problème | Fix | Fichier |
|---|----------|-----|---------|
| 1 | `AttributeError: 'float' has no attribute 'question'` dans `flatten()` — NaN passe `if mcq` | `hasattr(mcq, 'question')` | `notebooks/generate_mcq.py` |
| 2 | `TypeError: Expected numeric dtype` sur `distractor_quality.round(0)` — GPT-4o renvoie parfois des strings | `pd.to_numeric(..., errors='coerce')` | `src/main.py` |
| 3 | `OSError: Repo id must be in the form 'repo_name/...'` pour openbiollm | Supprimé `hf.co/` et `:latest` | `notebooks/generate_mcq.py` |
| 4 | Pipeline HF rechargé à chaque sheet (1592 chargements GPU par modèle) | `pipeline()` créé une fois dans `for_a_model()`, passé en paramètre | `notebooks/generate_mcq.py` |
| 5 | Checkpoint global (`start_`, `df_in_construction_.csv`) — un modèle peut hériter du checkpoint d'un autre | Checkpoints nommés par modèle (`start_{model}`, `df_in_construction_{model}.csv`) | `notebooks/generate_mcq.py` |
| 6 | `compare_systems.sh` faisait tous-old puis tous-new — si crash, reprise difficile | Boucle par modèle : old puis new pour chaque modèle. Backup CSVs dans `csv_mcq_{old,new}/` | `src/compare_systems.sh` |

### Structure de `comparison_results/` après un run complet

```
comparison_results/
  state_old.txt / state_new.txt    ← modèles complétés (reprise automatique)
  distribution_old.output          ← résultats agrégés système old
  distribution_new.output          ← résultats agrégés système new
  csv_mcq_old/ csv_mcq_new/        ← backups CSVs générés
  csv_eval_old/ csv_eval_new/      ← backups CSVs évalués
```

### Reset partiel

```bash
# Reset un modèle sur les deux systèmes
sed -i '/^gemma2_9b$/d' comparison_results/state_old.txt
sed -i '/^gemma2_9b$/d' comparison_results/state_new.txt

# Reset complet
rm comparison_results/state_*.txt
```

---

## Prompt Auto-Optimization (2026-04-30 → en cours)

**Hypothèse :** la qualité médiocre des distracteurs n'est pas une limite du modèle mais du prompt.
Un prompt mieux formulé → meilleures données d'entraînement → meilleur modèle finetuné.

**Approche :** boucle de feedback itérative par modèle, Claude (API) comme optimiseur de prompt.

### Algorithme v1 (implémenté)

```
prompt = prompt_initial

POUR chaque fiche LISA s dans k_fiches:
    POUR tentative in 1..max_attempts:
        mcq   = model.generate(prompt, s)
        b3    = GPT-4o.eval_distractor_quality(mcq, s)
        SI b3.passe (≥2/3 distracteurs avec score ≥4):
            break → fiche suivante
        SINON si tentative < max_attempts:
            prompt = Claude.improve(prompt, {mcq, b3_scores, b3_justifications})

SAUVEGARDER prompt final pour ce modèle
```

### Algorithme v2 — Rollback (implémenté, superseded by v3)

Problème de v1 : le prompt est toujours mis à jour même quand la fiche échoue toutes ses tentatives,
accumulant des modifications contradictoires qui dégradent les fiches suivantes.

Rollback implémenté mais la boucle restait single-pass (question + distracteurs couplés).
**Remplacé par l'algorithme v3 deux étapes** qui apporte le même rollback sur deux signaux indépendants.

### Algorithme v3 — Two-stage + Rollback (implémenté 2026-05-19)

Signal mixte de v1/v2 (question + distracteurs dans la même boucle) → oscillations.
Solution : deux boucles indépendantes avec rollbacks séparés, puis fusion d'un prompt unique final.

```
prompt_q = _build_initial_question_prompt()
prompt_d = _build_initial_distractor_prompt()

POUR chaque fiche LISA s dans k_fiches:
    prompt_q_entree = prompt_q          ← snapshot rollback question
    prompt_d_entree = prompt_d          ← snapshot rollback distracteurs

    # Stage 1 : question
    question_validated = False
    POUR q_attempt in 1..max_q_attempts:
        step1 = generate_question(s, prompt_q, pipe)   ← MCQStep1 (question + correct_answer)
        qresult = evaluate_question_quality(step1.question, s)
            # A1 is_question, A2 no_negation, A3 originality≥0.75, A4 FK≥12, B1 cosine≥0.5
        SI qresult.passes:
            question_validated = True; break
        SINON si q_attempt < max_q_attempts:
            prompt_q = improve_question_prompt(prompt_q, step1, qresult)
            # Qwen3.5-397B améliore uniquement <<<DEBUT_REGLES_QUESTION>>>…<<<FIN_REGLES_QUESTION>>>

    SI NOT question_validated:
        prompt_q = prompt_q_entree      ← rollback Q
        continue → fiche suivante

    # Stage 2 : distracteurs (question gelée)
    distractor_passed = False
    POUR d_attempt in 1..max_d_attempts:
        step2 = generate_distractors(s, step1.question, step1.correct_answer, prompt_d, pipe)
        mcq   = assemble_mcq(step1, step2)   ← slots a-d mélangés aléatoirement
        b3    = GPT-4o.eval_distractor_quality(mcq, s)
        SI b3.passes (≥2/3 distracteurs score ≥4):
            distractor_passed = True; break
        SINON si d_attempt < max_d_attempts:
            prompt_d = improve_distractor_prompt(prompt_d, step1, step2, b3)
            # Qwen améliore uniquement <<<DEBUT_REGLES_DISTRACTEURS>>>…<<<FIN_REGLES_DISTRACTEURS>>>

    SI NOT distractor_passed:
        prompt_d = prompt_d_entree      ← rollback D

# Fusion finale : un seul prompt déployable par modèle
final_prompt = build_final_prompt(prompt_q, prompt_d)
    # extrait les deux sections optimisées → prompt single-pass 11 champs
```

**Propriétés clés :**
- `prompt_q` et `prompt_d` évoluent indépendamment — le signal de chaque boucle est propre
- La question gelée entre Stage 1 et Stage 2 empêche de perdre une bonne question en itérant sur les distracteurs
- `final_prompt.txt` est un prompt standard 11 champs (déployable comme avant)

**Paramètres CLI :**
```bash
python scripts/optimize_prompt.py \
  --models mistral_7b \
  --k 20 \
  --max-q-attempts 3 \
  --max-d-attempts 5
```

### Modèles ciblés — ordre du plus petit au plus grand

| # | Save name | Modèle HuggingFace | Taille |
|---|-----------|-------------------|--------|
| 1 | qwen3_0_6b | Qwen/Qwen3-0.6B | 0.6B |
| 2 | mistral_7b | mistralai/Mistral-7B-Instruct-v0.3 | 7B |
| 3 | llama3_8b | meta-llama/Llama-3.1-8B-Instruct | 8B |
| 4 | openbiollm_8b | antonkirk/Llama3-Instruct-OpenBioLLM-8B-merged | 8B |
| 5 | apertus_8b | swiss-ai/Apertus-8B-Instruct-2509 | 8B |
| 6 | gemma2_9b | google/gemma-2-9b-it | 9B |
| 7 | eurollm_9b | utter-project/EuroLLM-9B-Instruct | 9B |
| 8 | magistral_small | mistralai/Magistral-Small-2509 | ~24B |
| 9 | medgemma_27b | google/medgemma-27b-it | 27B |
| 10 | nemotron_30b | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 30B total (3B actifs, MoE) |
| 11 | gemma4_31b | google/gemma-4-31B-it | 31B |
| 12 | qwen3_5_35b | Qwen/Qwen3.5-35B-A3B | 35B total (3B actifs, MoE) |
| 13 | mixtral_8x7b | mistralai/Mixtral-8x7B-Instruct-v0.1 | ~47B total (14B actifs, MoE) |

### Paramètres

| Param | Valeur défaut |
|-------|--------------|
| k (fiches LISA) | 20 |
| max_attempts / fiche | 5 |
| Critère B3 | ≥ 2 distracteurs sur 3 avec score ≥ 4/5 |
| Optimiseur | Claude Opus (`claude-opus-4-7`) via OpenRouter |

### Implémentation

- **Script** : `scripts/optimize_prompt.py`
- **Réutilise** : pipeline HuggingFace (génération), `src/eval/distractors_quality.py` (B3), `src/eval/prompts.json`
- **Output** : `data/optimized_prompts/{model_name}/final_prompt.txt` + `optimization_log.json`
- **Trace** : `data/optimized_prompts/trace.jsonl` (événement par événement, streamable avec `tail -f`)
- **Visualisation** : `scripts/plot_optimization.py` → `data/optimized_prompts/plot_b3.png`

### Bugs détectés et corrigés (runs OAR 123947 & 123975)

| # | Problème | Cause | Fix |
|---|----------|-------|-----|
| 1 | `+0 chars` sur ~90% des modifications (job 123947) | Claude ne respectait pas les balises `<<<PROMPT_DEBUT>>>` / `<<<PROMPT_FIN>>>` quand l'historique s'allongeait — prompt extrait brut, sentinel absent → prompt conservé inchangé | Corrigé entre jobs : balises présentes dans job 123975 |
| 2 | Coupures massives du prompt par Claude (-27k, -46k, -32k chars) | Aucune contrainte de taille dans le message à Claude — il réécrit depuis zéro sur les fiches difficiles | Ajout d'une contrainte dynamique `[70%–150%]` de la longueur courante dans chaque appel |
| 3 | Modifications en dehors de la section distracteurs | Pas de scope défini — Claude modifiait aussi le stem, le JSON, les instructions générales | Ajout dans `_IMPROVE_SYSTEM` : « modifier UNIQUEMENT la section relative aux règles des distracteurs » |
| 4 | Pas de garde-fou post-extraction | Claude ignore parfois les contraintes — coupures > 30% passaient quand même | Guard dans `improve_prompt()` : si `len(improved) / len(before) < 0.7` → rollback silencieux |

### Résultats observés (medgemma_27b, 20 fiches)

| Job OAR | Taux de réussite B3 | Notes |
|---------|---------------------|-------|
| 123947 | 5/20 (25%) | Modifications Claude silencieuses (+0 chars) dès fiche 3 |
| 123975 | 3/15 vues (20%, en cours) | Modifications effectives mais oscillations violentes |

**Observation clé :** la boucle actuelle (v1) ne converge pas — le score moyen des distracteurs
n'augmente pas entre les tentatives sur les fiches difficiles, et les modifications successives se
contredisent.

### Visualisation de la convergence

`scripts/plot_optimization.py` génère une courbe par modèle :
- **X** : index de tentative cumulatif (toutes fiches confondues, dans l'ordre)
- **Y** : score moyen des 3 distracteurs (1–5)
- Points verts = B3 passé, points rouges = B3 échoué
- Ligne bleue = moyenne glissante (fenêtre 5)
- Tirets gris = séparations de fiches
- Annotations orange = coupures Claude > 5 000 chars

---

## Open Questions

- [ ] Does the reasoning trace improve distractor quality vs. non-reasoning models on LISA content?
- [ ] Should the `thinking` trace be stored alongside MCQs (for auditing / quality review)?
- [ ] Is 80B Qwen3 fast enough for interactive generation, or only for batch jobs?
- [ ] Can the thinking trace be used as a quality signal to auto-filter bad MCQs?
- [ ] What are the empirical bounds for the ambiguity window on this French medical corpus?
- [ ] Does prompt optimization generalize across LISA sheets (held-out test set)?
- [ ] How many iterations does Claude need on average to pass B3 per model size?
- [ ] Le rollback améliore-t-il le taux de réussite B3 par rapport à v1 ?
- [ ] La courbe de score moyen des distracteurs converge-t-elle vers le haut avec suffisamment de fiches ?

---

## Architecture v3 — Two-stage (implémentée 2026-05-19)

Anciennement "Perspectives futures — Architecture alternative". Implémentée dans `scripts/optimize_prompt.py`.

### Schémas Pydantic (nouveaux)

```python
class MCQStep1(BaseModel):
    question: str
    question_comment: Optional[str] = ""
    correct_answer: str

class MCQStep2(BaseModel):
    distractor_1: str; distractor_1_comment: Optional[str] = ""
    distractor_2: str; distractor_2_comment: Optional[str] = ""
    distractor_3: str; distractor_3_comment: Optional[str] = ""
```

`assemble_mcq(step1, step2) → MCQQuestion` : mélange aléatoire des slots a-d.

### Structure des prompts avec balises de section

```
prompt_q :
  <<<DEBUT_REGLES_QUESTION>>>
  [règles question — modifiées par Qwen si A1/A2/A3/A4/B1 échouent]
  <<<FIN_REGLES_QUESTION>>>
  → JSON 3 champs : question, question_comment, correct_answer

prompt_d :
  Question : <<<QUESTION>>>  Réponse : <<<REPONSE_CORRECTE>>>  Contenu : <<<CONTENU_EDUCATIF>>>
  <<<DEBUT_REGLES_DISTRACTEURS>>>
  [règles distracteurs — modifiées par Qwen si B3 échoue]
  <<<FIN_REGLES_DISTRACTEURS>>>
  → JSON 6 champs : distractor_1/2/3 + comments
```

`build_final_prompt(prompt_q, prompt_d)` extrait les deux sections et produit un prompt single-pass 11 champs.

### Métriques question (Stage 1) — toutes locales, sans GPT-4o

| Id | Métrique | Seuil | Implémentation |
|----|----------|-------|----------------|
| A1 | is_question | True | `eval/question_check.py` |
| A2 | no negation | True | `eval/negation.py` (anglais uniquement — bug connu) |
| A3 | originality | ≥ 0.75 | `eval/originality.py` (BGE cosine) |
| A4 | readability (FK) | ≥ 12 | `eval/readability.py` |
| B1 | relevance | ≥ 0.5 | `eval/relevance.py` (BGE cosine) |

### Dettes techniques

| # | Problème | Impact | Statut |
|---|----------|--------|--------|
| 1 | `starts_with_negation()` en anglais → A2 toujours True sur français | A2 inutile | Non bloquant, documenté |
| 2 | `render_trace.py` ne parse pas `question_quality` ni `stage` | HTML trace partiel | Non bloquant |
| 3 | VRAM : BGE (~440 MB) coexiste avec SLM sur GPU | Risque OOM sur 27B+ | À surveiller |

---

## Next Steps (incremental)

1. ~~Run both models on a real LISA sheet prompt and compare MCQ output quality~~ → benchmark script done
2. ~~Fix evaluation pipeline issues (embedding model, context, position bias, raw scores)~~ → done
3. ~~Implémenter `scripts/optimize_prompt.py`~~ → done (v1)
4. ~~Implémenter le rollback + architecture deux étapes (question gelée)~~ → done (v3, 2026-05-19)
5. **Lancer un run complet sur les 13 modèles** avec la v3 et comparer le taux B3 à la v1
6. Analyser `plot_b3.png` : vérifier que la moyenne glissante monte sur les premiers modèles
7. Run reliability tests (`reliability_test.py`) avant d'interpréter les résultats de benchmark
8. Implement ambiguity window `[0.3 – 0.75]` once calibration data is available
9. Benchmark latency: magistral 24B vs Nemotron 30B
10. Decide whether to store `thinking` trace in the custom MCQ JSON schema

---

## References

- Ollama chat API: `POST /api/chat` with `think: true`
- Custom MCQ pipeline: `src/page/backend/api/routes/generation.py`
- Custom MCQ schema: `src/page/backend/data/custom_mcqs/*.json`
- Eval pipeline: `src/page/backend/eval/`
- Haladyna & Rodriguez (2013) — *Developing and Validating Test Items*, Routledge
- Gierl et al. (2017) — *Review of Educational Research* 87(6)
- Mitkov, Ha & Karamanis (2009) — ACL Workshop
- Yeung & Lee et al. (2019) — ALTA
