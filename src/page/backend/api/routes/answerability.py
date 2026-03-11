"""
Answerability Routes - Evaluate a model's ability to answer MCQs from its own knowledge.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, ConfigDict

from ..utils.dependencies import get_current_user
from ..models.auth import User

router = APIRouter()

BACKEND_DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_FILE = BACKEND_DATA_DIR / "answerability" / "results.json"

# Path to the sensitive UNESS dataset — fixed server-side, never derived from user input
_UNESS_FILE = Path(__file__).resolve().parents[5] / "data" / "mcqs_answerability.json"

# Columns used by both UNESS and the preprocessed MedMCQA
_INTERNAL_COLS = dict(
    question_col="question",
    option_a_col="opa",
    option_b_col="opb",
    option_c_col="opc",
    option_d_col="opd",
    correct_col="cop",
)

# In-memory job store: {job_id: {status, progress, total, accuracy, correct, error}}
_jobs: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# MODELS
# ============================================================================

class AnswerabilityRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str               # "ollama" | "hf"
    model_name: str
    dataset_name: str             # "uness" | HuggingFace id e.g. "openlifescienceai/medmcqa"
    dataset_split: str = "train"
    preprocess_mode: str = "none" # "none" | "medmcqa_stratified"
    question_col: str = "question"
    option_a_col: str = "opa"
    option_b_col: str = "opb"
    option_c_col: str = "opc"
    option_d_col: str = "opd"
    correct_col: str = "cop"
    sample_size: Optional[int] = None


# ============================================================================
# PERSISTENCE
# ============================================================================

def _load_results() -> list:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_result(result: dict):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = _load_results()
    # Keep only the latest result per (model_name, dataset_name) pair
    key = (result["model_name"], result["dataset_name"])
    results = [r for r in results if (r["model_name"], r["dataset_name"]) != key]
    results.append(result)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ============================================================================
# DATASET LOADING
# ============================================================================

def _load_uness_df():
    """Load the UNESS dataset from the fixed internal path."""
    if not _UNESS_FILE.exists():
        raise FileNotFoundError(
            f"Fichier UNESS introuvable : {_UNESS_FILE}. "
            "Vérifiez que le fichier data/mcqs_answerability.json est bien présent à la racine du projet."
        )
    import pandas as pd
    return pd.read_json(_UNESS_FILE)


def _load_medmcqa_stratified_df():
    """
    Load MedMCQA with the same preprocessing as the notebook:
    - Concatenate train + test + validation splits
    - Stratified sample: max 200 per subject_name (exclude 'Unknown')
    """
    from datasets import load_dataset, concatenate_datasets
    ds = load_dataset("openlifescienceai/medmcqa")
    all_splits = concatenate_datasets([ds["train"], ds["test"], ds["validation"]])
    df = all_splits.to_pandas()
    df_grouped = df.groupby("subject_name")
    df_stratified = (
        df_grouped
        .apply(lambda g: g.sample(n=min(200, len(g)), random_state=42) if g.name != "Unknown" else g.iloc[0:0])
        .reset_index(drop=True)
    )
    return df_stratified


# ============================================================================
# INFERENCE HELPERS
# ============================================================================

SYSTEM_PROMPT = (
    "You are tasked with answering multiple-choice questions, "
    "containing 4 different answer options - a, b, c and d.\n"
    "Provide just a single letter corresponding to the correct option as the response.\n"
    "For example: \"a\", \"b\", \"c\", or \"d\""
)


def _generate_mcq_prompt(row, question_col, option_a_col, option_b_col, option_c_col, option_d_col) -> str:
    return (
        f"-----\n"
        f"Question:\n{row[question_col]}\n"
        f"Options:\n"
        f"a) {row[option_a_col]}\n"
        f"b) {row[option_b_col]}\n"
        f"c) {row[option_c_col]}\n"
        f"d) {row[option_d_col]}\n"
        f"-----"
    )


def _parse_letter(text: str) -> str:
    """Extract a/b/c/d from a model response."""
    idx = text.find(")")
    if idx > 0 and text[idx - 1].lower() in "abcd":
        return text[idx - 1].lower()
    for c in text.strip().lower():
        if c in "abcd":
            return c
    raise ValueError(f"Cannot parse answer from: {text!r}")


def _answer_ollama(mcq_text: str, model_name: str) -> str:
    from ollama import chat
    response = chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": mcq_text},
        ],
        options={"temperature": 0.1, "num_ctx": 2048},
    )
    return _parse_letter(response["message"]["content"])


def _answer_hf(mcq_text: str, outlines_model) -> str:
    from typing import Literal
    response = outlines_model(
        f"{SYSTEM_PROMPT}\n\n{mcq_text}",
        Literal["a", "b", "c", "d"]
    )
    return response.lower()


def _normalize_correct(val) -> int:
    """Convert correct answer to int index 0-3."""
    try:
        return int(val)  # handles int, float, numpy.int64/float64, and numeric strings
    except (ValueError, TypeError):
        pass
    mapping = {"a": 0, "b": 1, "c": 2, "d": 3}
    return mapping.get(str(val).lower().strip(), -1)


# ============================================================================
# BACKGROUND JOB
# ============================================================================

def _run_answerability(job_id: str, req: AnswerabilityRequest):
    job = _jobs[job_id]
    try:
        job["status"] = "loading_dataset"

        # ---- Internal datasets (never derived from user path input) ----
        if req.dataset_name == "uness":
            df = _load_uness_df()
            q_col  = _INTERNAL_COLS["question_col"]
            oa_col = _INTERNAL_COLS["option_a_col"]
            ob_col = _INTERNAL_COLS["option_b_col"]
            oc_col = _INTERNAL_COLS["option_c_col"]
            od_col = _INTERNAL_COLS["option_d_col"]
            cop_col = _INTERNAL_COLS["correct_col"]

        # ---- MedMCQA with notebook-style preprocessing ----
        elif req.preprocess_mode == "medmcqa_stratified":
            df = _load_medmcqa_stratified_df()
            q_col, oa_col, ob_col, oc_col, od_col, cop_col = (
                req.question_col, req.option_a_col, req.option_b_col,
                req.option_c_col, req.option_d_col, req.correct_col,
            )

        # ---- Generic HuggingFace dataset ----
        else:
            from datasets import load_dataset
            dataset = load_dataset(req.dataset_name, split=req.dataset_split)
            df = dataset.to_pandas()
            q_col, oa_col, ob_col, oc_col, od_col, cop_col = (
                req.question_col, req.option_a_col, req.option_b_col,
                req.option_c_col, req.option_d_col, req.correct_col,
            )

        # Check cancellation after dataset loading (can be long for MedMCQA)
        if job.get("cancelled"):
            job["status"] = "cancelled"
            job["error"] = "Annulé par l'utilisateur."
            return

        # Optional random sub-sample
        if req.sample_size and req.sample_size < len(df):
            df = df.sample(n=req.sample_size, random_state=42).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        job["total"] = len(df)

        # ---- Load HF model if needed ----
        outlines_model = None
        if req.model_type == "hf":
            job["status"] = "loading_model"
            import outlines
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(
                req.model_name, use_fast=True, trust_remote_code=True
            )
            outlines_model = outlines.from_transformers(
                AutoModelForCausalLM.from_pretrained(req.model_name, device_map="auto"),
                tokenizer,
            )
            # Check cancellation after model loading (peut prendre plusieurs minutes)
            if job.get("cancelled"):
                job["status"] = "cancelled"
                job["error"] = "Annulé par l'utilisateur."
                return

        # ---- Evaluate ----
        job["status"] = "running"
        conv = {"a": 0, "b": 1, "c": 2, "d": 3}
        correct_count = 0

        for idx in range(len(df)):
            # Check cancellation flag before each question
            if job.get("cancelled"):
                job["status"] = "cancelled"
                job["error"] = "Annulé par l'utilisateur."
                print(f"🛑 Answerability cancelled — {job_id} at {idx}/{len(df)}")
                return

            row = df.iloc[idx]
            mcq_text = _generate_mcq_prompt(row, q_col, oa_col, ob_col, oc_col, od_col)

            generated = None
            for attempt in range(5):
                try:
                    if req.model_type == "ollama":
                        letter = _answer_ollama(mcq_text, req.model_name)
                    else:
                        letter = _answer_hf(mcq_text, outlines_model)
                    generated = conv[letter]  # KeyError si lettre invalide → retry
                    break
                except (SyntaxError, KeyError) as e:
                    print(f"[answerability] SyntaxError ou KeyError, relance... (attempt {attempt+1})")
                    if attempt == 4:
                        generated = None
                except Exception as e:
                    print(f"[answerability] attempt {attempt+1} failed for idx {idx}: {e}")

            if generated is not None:
                cop = _normalize_correct(row[cop_col])
                if generated == cop:
                    correct_count += 1

            job["progress"] = idx + 1
            job["correct"] = correct_count

        accuracy = round(correct_count / len(df) * 100, 2) if len(df) > 0 else 0.0
        job["status"] = "done"
        job["accuracy"] = accuracy

        _save_result({
            "job_id": job_id,
            "model_name": req.model_name,
            "model_type": req.model_type,
            "dataset_name": req.dataset_name,
            "sample_size": len(df),
            "accuracy": accuracy,
            "correct": correct_count,
            "date": datetime.utcnow().isoformat(),
        })

        print(f"✅ Answerability done — {job_id}: {accuracy:.2f}% ({correct_count}/{len(df)})")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"❌ Answerability failed — {job_id}: {e}")


# ============================================================================
# ROUTES
# ============================================================================

@router.get("/answerability/models/ollama")
async def list_ollama_models(current_user: User = Depends(get_current_user)):
    """List available Ollama models."""
    try:
        import ollama
        result = ollama.list()
        names = [m["model"] for m in result.get("models", [])]
        return {"models": names}
    except Exception as e:
        return {"models": [], "error": str(e)}


@router.get("/answerability/results")
async def get_results(current_user: User = Depends(get_current_user)):
    """Return the latest answerability result per (model, dataset)."""
    return {"results": _load_results()}


@router.post("/answerability/run")
async def run_answerability(
    req: AnswerabilityRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """Start an answerability evaluation job."""
    job_id = f"ANS-{uuid.uuid4().hex[:7].upper()}"
    _jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "total": 0,
        "accuracy": None,
        "correct": 0,
        "error": None,
        "cancelled": False,
    }
    background_tasks.add_task(_run_answerability, job_id, req)
    print(f"🚀 Answerability job started — {job_id} (model: {req.model_name}, dataset: {req.dataset_name})")
    return {"job_id": job_id}


@router.get("/answerability/job/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Poll the status of an answerability job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable.")
    return {"job_id": job_id, **job}


@router.delete("/answerability/job/{job_id}")
async def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Request cancellation of a running job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable.")
    if job["status"] not in ("pending", "loading_dataset", "loading_model", "running"):
        raise HTTPException(status_code=400, detail=f"Le job est déjà terminé (statut: {job['status']}).")
    job["cancelled"] = True
    print(f"🛑 Cancellation requested — {job_id}")
    return {"job_id": job_id, "message": "Annulation demandée."}
