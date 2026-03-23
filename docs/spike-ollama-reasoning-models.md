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

## Evaluation Pipeline — Scientific Review & Corrections

Following a scientific review of the evaluation approach (conducted from the perspective of a senior psychometrician and medical examiner), three metrics were identified as problematic: **Distractor Quality**, **Ambiguity**, and **Relevance**.

### Metrics with issues

#### Distractor Quality (B3) — 3 problems fixed
| Problem | Fix |
|---------|-----|
| Binarisation prématurée : un seul booléen masquait les scores individuels — un distractor catastrophique [4,4,1] passait inaperçu | `distractors_quality.py` stocke désormais les scores bruts + avg dans `distractors_quality_detail`. La card B3 affiche `Rang X \| avg=X.XX \| scores=[x,x,x]` |
| GPT-4o évaluait sans accès au contenu LISA — ne pouvait pas juger la pertinence médicale des misconceptions | `context_col='content_raw'` passé à `generate_prompt_for_question()` |
| Seuil fixe `≥ 4` indépendant du Rang LISA — pénalisait injustement les items Rang A (connaissance de base) | Seuil adaptatif : `≥ 3` pour Rang A (éviter la variance construct-irrelevante), `≥ 4` pour Rang B (raisonnement discriminant). Règle additionnelle : `min(scores) ≥ 2` (aucun distractor catastrophique). |

**Bibliographic justification for rang-adaptive threshold:** Haladyna & Rodriguez (2013), Gierl et al. (2017, *Review of Educational Research*), Bloom's taxonomy levels 1–2 vs 3–4.

#### Ambiguity (B6) — 2 problems, 1 fixed
| Problem | Status |
|---------|--------|
| Modèle d'embedding `BAAI/bge-base-en-v1.5` anglais sur corpus français — qualité dégradée | ✓ Remplacé par `almanach/moderncamembert-base` |
| Seuil unique `≥ 0.3` : ne capture que la borne basse (distractor trivial). La borne haute (distractor trop similaire = ambigu) manque | ⏳ Fenêtre `[0.3 – 0.75]` à implémenter |

**Bibliographic justification for window:** Mitkov et al. (2009) — intermediate similarity → highest item discrimination; Yeung & Lee et al. (2019) — `STS < 80%` upper filter; ArXiv 2025 (Student Choice Prediction) — best distractors cluster at intermediate cosine similarity.

#### Relevance (B2) — 1 problem fixed
| Problem | Fix |
|---------|-----|
| Même modèle d'embedding anglais que Ambiguity | ✓ Remplacé par `almanach/moderncamembert-base` |

### Other fixes

| # | Issue | Fix |
|---|-------|-----|
| Position bias | Generation model places correct answer in position `a`; GPT-4o over-scores first distractor | `shuffle_distractors()` added in `generate_mcq_GPU.py`, called after every `validate_mcq()` |
| Answerability absent de la card | Métrique calculée mais non affichée | Added as **B5** in `section_b_checks` |
| Ambiguity absent de la card | Métrique calculée mais non affichée | Added as **B6** in `section_b_checks` |

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

## Benchmark Scripts

### benchmark_nemotron_vs_magistral.py (legacy)
Compares **Nemotron 30B** vs **Magistral 24B** on 10 LISA rows.

### benchmark_thinking_vs_nothinking.py (new — 2026-03-13)
Compares **6 thinking/non-thinking model pairs** on **1000 LISA rows**. All metrics evaluated simultaneously.

```bash
cd notebooks
python benchmark_thinking_vs_nothinking.py
# → docs/benchmark_thinking_vs_nothinking.html
# Checkpoints in docs/checkpoints/ — resumable on crash
```

**Model pairs:**

| Pair | Thinking | Non-thinking | Mechanism |
|------|----------|--------------|-----------|
| Qwen3 14B | `qwen3:14b` (think=True) | `qwen3:14b` (think=False) | Same weights, Ollama `think` flag |
| Mistral ~24B | `magistral:latest` (think=True) | `mistral-small3.1:24b` | Same family |
| Nemotron 30B | `Nemotron-3-Nano-30B` | `Nemotron-3-Nano-30B` | Internal `<think>` tags |
| Phi-4 14B | `phi4-reasoning:14b` | `phi4:14b` | Same base, R1-style SFT |
| Qwen2.5 14B | `deepseek-r1:14b` | `qwen2.5:14b` | Same weights, R1 distillation |
| Qwen2.5 32B | `qwq:32b` | `qwen2.5:32b` | Same family |

**Key finding — Ollama `think` flag support:**

| Model | think=True via /api/chat | Fallback |
|-------|--------------------------|---------|
| `qwen3:14b` | ✓ supported | — |
| `magistral:latest` | ✓ supported | — |
| `nemotron-30b` | ✗ HTTP 400 | Uses internal `<think>` tags stripped by `extract_json()` |
| `phi4-reasoning:14b` | ✗ HTTP 400 | Same |
| `deepseek-r1:14b` | ✗ HTTP 400 but generates fine | Same |
| `qwq:32b` | ✗ HTTP 400 | Same |

→ Added `generate_mcq_chat(prompt, model, think: bool)` to `generate_mcq_GPU.py` — uses `/api/chat` instead of `/api/generate`, controls thinking via Ollama flag.

**B6 Ambiguity** set to informative-only (`result: N/A`) pending empirical calibration of the `[0.3–0.75]` window on French medical corpus.

---

## Open Questions

- [ ] Does the reasoning trace improve distractor quality vs. non-reasoning models on LISA content?
- [ ] Should the `thinking` trace be stored alongside MCQs (for auditing / quality review)?
- [ ] Is 80B Qwen3 fast enough for interactive generation, or only for batch jobs?
- [ ] Can the thinking trace be used as a quality signal to auto-filter bad MCQs?
- [ ] What are the empirical bounds for the ambiguity window on this French medical corpus?

---

## Next Steps (incremental)

1. ~~Run both models on a real LISA sheet prompt and compare MCQ output quality~~ → done
2. ~~Fix evaluation pipeline issues (embedding model, context, position bias, raw scores)~~ → done
3. ~~Build thinking vs non-thinking benchmark (6 pairs, 1000 LISA rows)~~ → script ready, end-to-end tested on 1 row
4. **Run full benchmark** (`N_LISA_ROWS=1000`) — set overnight
5. Run reliability tests (`reliability_test.py`) before interpreting benchmark results
6. Implement ambiguity window `[0.3 – 0.75]` once calibration data is available
7. Decide whether to store `thinking` trace in the custom MCQ JSON schema

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
