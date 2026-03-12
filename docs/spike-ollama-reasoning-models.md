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

## Open Questions

- [ ] Does the reasoning trace improve distractor quality vs. non-reasoning models on LISA content?
- [ ] Should the `thinking` trace be stored alongside MCQs (for auditing / quality review)?
- [ ] Is 80B Qwen3 fast enough for interactive generation, or only for batch jobs?
- [ ] Can the thinking trace be used as a quality signal to auto-filter bad MCQs?
- [ ] What are the empirical bounds for the ambiguity window on this French medical corpus?

---

## Next Steps (incremental)

1. ~~Run both models on a real LISA sheet prompt and compare MCQ output quality~~ → benchmark script done
2. ~~Fix evaluation pipeline issues (embedding model, context, position bias, raw scores)~~ → done
3. Run reliability tests (`reliability_test.py`) before interpreting benchmark results
4. Implement ambiguity window `[0.3 – 0.75]` once calibration data is available
5. Benchmark latency: magistral 24B vs Nemotron 30B
6. Decide whether to store `thinking` trace in the custom MCQ JSON schema

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
