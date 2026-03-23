# Distractor Plausibility — Paper Excerpt & References

## Paragraph

**Distractor Plausibility.** Are distractors realistic and confusing in a pedagogically meaningful way? To compute this, we compare distractors two by two against the correct answer, asking the LLM to provide a brief justification for each plausibility score to ensure the ratings are not random. We then apply a majority vote to confirm whether the question's distractors are considered plausible.

A known confound in LLM-based distractor evaluation is position bias: generation models systematically place the correct answer in the first position, and judge models tend to over-score the first option presented (Ko et al., 2020; Zheng et al., 2023). To neutralize this, we randomly shuffle the four options before submitting each item to the judge, updating the `correct_option` label accordingly. This ensures that plausibility scores reflect semantic quality rather than positional priming.

Even if the majority vote passes, we do not want the minority distractor to be trivially eliminable by students, as this would undermine item discrimination. We therefore add the constraint that every individual distractor score must be at least 2 — ensuring no single distractor is catastrophically implausible regardless of the majority outcome.

Finally, we apply a **rang-adaptive threshold** on the majority vote average. The LISA dataset classifies each learning objective into Rang A (foundational knowledge) or Rang B (applied reasoning). These ranks map directly onto Bloom's taxonomy levels: Rang A items target recall and comprehension (levels 1–2), while Rang B items target application and analysis (levels 3–4). Accordingly, the plausibility bar is raised for higher-order items: for Rang A, distractors are required to reach an average score of **≥ 3** — they must represent plausible misconceptions at the knowledge level, but need not discriminate finely between subtly different clinical reasoning paths. For Rang B, the threshold is **≥ 4** — distractors must be sufficiently similar to the correct answer to challenge students who have partial but incomplete understanding, as required by items targeting higher cognitive demand. This design choice is grounded in Haladyna & Rodriguez (2013), who show that distractor quality criteria should be commensurate with the cognitive level of the item, and in Gierl et al. (2017), who demonstrate that nonfunctional distractors disproportionately degrade discrimination on higher-order items.

---

## Answerability

**Answerability.** Does the question have a single, unambiguous correct answer that a knowledgeable reader can identify from the options provided? To assess this, we submit the question and its four options — along with the LISA source content as context — to GPT-4o and ask it to select the correct answer. The predicted answer is then compared to the ground-truth `correct_option`. A match indicates that the question is well-formed and that the correct answer is indeed distinguishable from the distractors. A mismatch signals either a flawed question stem, an incorrect ground-truth label, or distractors that are insufficiently differentiated from the correct answer.

---

## Ambiguity

**Ambiguity.** Are any distractors so semantically close to the correct answer that they could be mistaken for it? For each distractor, we compute the cosine similarity between its embedding and the embedding of the correct answer using ModernCamemBERT. We then average these similarities across the three distractors to obtain a single ambiguity score per item. A high score indicates that the distractor set clusters near the correct answer in semantic space, which risks confusing well-prepared students and degrading item validity. Following Mitkov et al. (2009) and Yeung & Lee et al. (2019), optimal distractors occupy an intermediate similarity window — close enough to be plausible, but distinct enough not to be interchangeable with the correct answer. The precise bounds of this window on French medical content are not yet empirically established; the metric is therefore reported as informative only, pending calibration on a representative sample.

---

## Reliability

> ⚠️ *Work in progress — reliability experiments not yet conducted. The methodology below is defined and the script is implemented (`notebooks/reliability_test.py`); results will be added once runs are complete.*

**Intra-rater reliability.** To assess the stability of the LLM judge, we submit the same MCQ twice to GPT-4o under identical conditions and measure the variance in distractor plausibility scores across the two runs. We report the mean absolute difference and standard deviation per distractor, as well as the PASS/FAIL agreement rate. A mean absolute difference below 0.5 points is considered acceptable, following the convention for human rater consistency in educational measurement (Stemler, 2004). High variance would indicate that scores are driven by stochastic sampling rather than genuine quality assessment, which would undermine the validity of the metric.

**Inter-rater reliability.** To validate the LLM judge against human judgment, a subset of 30–50 MCQs is annotated manually by a domain expert using the same 1–5 plausibility scale. We then compute linear Cohen's κ between the GPT-4o scores and the human annotations. A κ ≥ 0.6 is targeted, corresponding to substantial agreement (Landis & Koch, 1977). This step is essential to confirm that the LLM judge captures the same construct as an expert medical examiner, rather than a proxy correlated with surface-level lexical features.

---

## Embedding Model

**Embedding model.** Both the Relevance and Ambiguity metrics rely on dense text representations computed via `almanach/moderncamembert-base` (ModernCamemBERT), a French encoder pre-trained on large-scale French corpora using a modernized BERT architecture. We deliberately chose a French-language encoder over multilingual or English-only alternatives (e.g. `BAAI/bge-base-en-v1.5`) to preserve the semantic structure of French medical terminology, which is systematically degraded by English-trained models.

---

## References

**Ko, W.-J., et al. (2020).** Inquisitive question generation for high level text comprehension. *EMNLP 2020.*

> Early evidence of positional preference in LLM evaluation settings.

---

**Zheng, L., et al. (2023).** Judging LLM-as-a-judge with MT-bench and Chatbot Arena. *NeurIPS 2023.*

> Demonstrates systematic position bias in GPT-4 evaluation: first-position options receive inflated scores. Motivates random shuffling before LLM-based scoring.

---

**Haladyna, T. M., & Rodriguez, M. C. (2013).** *Developing and Validating Test Items.* Routledge.

> The canonical reference in item writing. Chapter on distractor quality argues that plausibility criteria must be calibrated to the cognitive demand of the item — low-order items require recognizable misconceptions, high-order items require distractors that challenge partial understanding.

---

**Gierl, M. J., Bulut, O., Guo, Q., & Zhang, X. (2017).** Developing, analyzing, and using distractors for multiple-choice tests in education: A comprehensive review. *Review of Educational Research, 87*(6), 1082–1116.

> Systematic review of distractor development methods. Key finding: nonfunctional distractors (never chosen, or chosen < 5% of examinees) disproportionately degrade item discrimination on higher-order items. Justifies raising the quality bar for Bloom L3-4 items.

---

**Bloom, B. S. (Ed.). (1956).** *Taxonomy of Educational Objectives: The Classification of Educational Goals. Handbook I: Cognitive Domain.* David McKay.

**Anderson, L. W., & Krathwohl, D. R. (Eds.). (2001).** *A Taxonomy for Learning, Teaching, and Assessing: A Revision of Bloom's Taxonomy of Educational Objectives.* Longman.

> Bloom levels 1–2 (recall, comprehension) map to Rang A; levels 3–4 (application, analysis) map to Rang B.

---

**Stemler, S. E. (2004).** A comparison of consensus, consistency, and measurement approaches to estimating interrater reliability. *Practical Assessment, Research & Evaluation, 9*(4).

> Defines acceptable intra-rater consistency thresholds for educational scoring contexts.

---

**Landis, J. R., & Koch, G. G. (1977).** The measurement of observer agreement for categorical data. *Biometrics, 33*(1), 159–174.

> Establishes the κ interpretation scale: < 0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, > 0.80 almost perfect. Target for LLM-as-judge validation: κ ≥ 0.6.

---

⚠️ Verify page numbers and volume/issue against your institution's database before submitting.
