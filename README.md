# Student Comment Classification with LLMs

## Project Overview
This project aims to **evaluate and track students' progress** using a QCM-based platform. Instead of just assessing their performance, we incorporate a **meta-cognitive layer** by asking students to:
1. **Evaluate their level of certainty** about their answers.
2. **Provide self-assessment comments** after reviewing the correct answers.

The goal is to **automatically classify these comments** using **Large Language Models (LLMs)** to enhance feedback and learning insights.

## Problem Statement
Given a set of inputs:
```json
{student, test, result, comment}
```
The model **M** must classify the student’s comment into a predefined category set:
```json
{C1, C2, ..., Cn}
```
producing an **enriched output**:
```json
{student, test, result, comment, C}
```
where **C** includes:
- **Global-Comment-Category**
- **Per-question classifications**: `{question, question-categories}`

## Approaches Considered

### 1️⃣ One-Shot LLM (RAG-Based)
- Uses a single **LLM with Retrieval-Augmented Generation (RAG)** to classify comments in one pass.

### 2️⃣ Orchestrator LLM
- Multiple specialized LLMs, each handling a **specific task**:
  - **Comment segmentation** into meaning units.
  - **Category assignment**.
  - **Evaluation and refinement**.
- An **Orchestrator LLM** manages communication between them to **ensure coherence**.

### 3️⃣ Debate LLMs
- Similar to the **Orchestrator LLM**, but instead of relying on strict orchestration, the LLMs **debate** among themselves to **agree on the best classification**.

### 4️⃣ Qvalue ... *(To be defined, blurred concept)*

## Next Steps
- Experiment with different architectures.
- Evaluate classification accuracy.
- Optimize feedback for students.

---

