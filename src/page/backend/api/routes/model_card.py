"""
Model Card Routes - Fetch HuggingFace model cards and extract relevant sections
using a local Ollama model as the extractor (Option B: LLM-based extraction).

Only sections with meaningful, non-boilerplate content are returned.
"""
import json
import os
import re
from fastapi import APIRouter, Depends, HTTPException
from ..utils.dependencies import get_current_user
from ..models.auth import User

router = APIRouter()

_EXTRACTOR_MODEL = os.getenv("MODEL_CARD_EXTRACTOR_MODEL", "mistral:7b-instruct")

# Mapping Ollama model names → HuggingFace model IDs
_OLLAMA_TO_HF: dict[str, str] = {
    # Qwen3
    "qwen3:0.6b":               "Qwen/Qwen3-0.6B",
    "qwen3:1.7b":               "Qwen/Qwen3-1.7B",
    "qwen3:4b":                 "Qwen/Qwen3-4B",
    "qwen3:8b":                 "Qwen/Qwen3-8B",
    "qwen3:14b":                "Qwen/Qwen3-14B",
    "qwen3:30b":                "Qwen/Qwen3-30B",
    "qwen3:32b":                "Qwen/Qwen3-32B",
    # Qwen2.5
    "qwen2.5:7b":               "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:14b":              "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5:32b":              "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5:72b":              "Qwen/Qwen2.5-72B-Instruct",
    # QwQ
    "qwq:32b":                  "Qwen/QwQ-32B",
    # Mistral
    "mistral:7b":               "mistralai/Mistral-7B-v0.1",
    "mistral:7b-instruct":      "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-small:24b":        "mistralai/Mistral-Small-3.1-24B",
    "magistral:24b":            "mistralai/Magistral-Small-24B",
    # Llama
    "llama3.1:8b":              "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1:70b":             "meta-llama/Llama-3.1-70B-Instruct",
    "llama3.2:3b":              "meta-llama/Llama-3.2-3B-Instruct",
    # DeepSeek
    "deepseek-r1:14b":          "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-r1:32b":          "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    # Phi
    "phi4:14b":                 "microsoft/phi-4",
    "phi4-reasoning:14b":       "microsoft/phi-4-reasoning",
    # Nvidia
    "nemotron:30b":             "nvidia/Nemotron-Mini-4B-Instruct",
    # Google
    "medgemma:4b":              "google/medgemma-4b",
    "medgemma:27b":             "google/medgemma-27b",
    "gemma2:9b":                "google/gemma-2-9b-it",
    # OpenBioLLM
    "openbiollm:8b":            "aaditya/OpenBioLLM-Llama3-8B",
    # EuroLLM
    "eurollm:9b":               "EuroLLM/EuroLLM-9B-Instruct",
}
_MAX_CARD_CHARS  = 12_000   # truncate very long cards before sending to LLM


EXTRACTION_PROMPT = """\
Tu analyses une model card HuggingFace (format markdown).

Extrais UNIQUEMENT les sections qui contiennent des informations significatives et spécifiques (pas de contenu générique ou boilerplate).
Retourne un objet JSON dont les clés sont les noms de sections et les valeurs sont des résumés concis (2-5 phrases) en français.

Utilise exactement ces clés lorsque le contenu correspondant est présent et pertinent :
- "usage_prévu"               : objectif principal, utilisateurs cibles, usages hors périmètre
- "méthode_entraînement"      : comment le modèle a été construit (pré-entraînement, fine-tuning, DPO, KTO, RLHF, SLERP, etc.)
- "données_entraînement"      : jeux de données, langues, domaines utilisés pour l'entraînement
- "données_évaluation"        : benchmarks et jeux de données utilisés pour l'évaluation
- "métriques"                 : métriques d'évaluation utilisées (accuracy, F1, BLEU, perplexité, etc.)
- "analyses_quantitatives"    : scores chiffrés sur benchmarks, résultats désagrégés
- "démarrage_rapide"          : comment utiliser le modèle (commandes, extrait de code, instructions)
- "facteurs"                  : groupes, conditions ou domaines influençant les performances
- "considérations_éthiques"   : biais, risques, potentiels détournements
- "mises_en_garde"            : limitations connues, faiblesses, recommandations

Règles STRICTES :
- Si une section est absente ou sans information concrète : NE PAS inclure la clé dans le JSON. Ne jamais écrire de phrase négative comme "pas de mention", "il n'y a pas", "aucune information", etc.
- Inclure une clé UNIQUEMENT si tu peux la remplir avec des faits précis tirés du texte.
- Résumés concis en français (2-5 phrases maximum).
- Retourner UNIQUEMENT du JSON valide — pas de balises markdown, pas de texte autour.

Contenu de la model card :
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_metadata(model_id: str) -> dict:
    from huggingface_hub import model_info
    try:
        info = model_info(model_id)
        card_data = getattr(info, "cardData", {}) or {}
        return {
            "license":             getattr(info, "license", None),
            "language":            list(card_data.get("language", []) or []),
            "pipeline_tag":        getattr(info, "pipeline_tag", None),
            "downloads_last_month": getattr(info, "downloads", None),
            "likes":               getattr(info, "likes", None),
            "tags":                list(getattr(info, "tags", []) or [])[:12],
        }
    except Exception:
        return {}


def _fetch_card_markdown(model_id: str) -> str:
    from huggingface_hub import ModelCard
    try:
        card = ModelCard.load(model_id)
        return card.content or ""
    except Exception:
        return ""


def _extract_via_ollama(card_markdown: str) -> dict:
    from ollama import Client
    client = Client(
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        timeout=120,
    )

    truncated = card_markdown[:_MAX_CARD_CHARS]
    if len(card_markdown) > _MAX_CARD_CHARS:
        truncated += "\n\n[...content truncated...]"

    response = client.chat(
        model=_EXTRACTOR_MODEL,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT + truncated}],
        options={"temperature": 0.0, "num_ctx": 8192},
    )

    raw = response.message.content.strip()
    # Strip <think>…</think> blocks (Qwen3 thinking mode)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Extract first JSON object
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    raw_sections = json.loads(m.group())
    return {
        k: v for k, v in raw_sections.items()
        if v and not v.strip().lower().startswith("il n'y a pas")
    }


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.get("/model-card/{model_id:path}")
async def get_model_card(
    model_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Fetch and extract relevant sections from a HuggingFace model card.
    Results are cached in memory for the lifetime of the server process.
    """
    model_id = model_id.strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")

    # Resolve Ollama names to HuggingFace IDs
    model_id = _OLLAMA_TO_HF.get(model_id.lower(), model_id)

    metadata = _fetch_metadata(model_id)
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found on HuggingFace or API unavailable.",
        )

    card_markdown = _fetch_card_markdown(model_id)

    sections: dict = {}
    extractor_error: str | None = None

    if card_markdown:
        try:
            sections = _extract_via_ollama(card_markdown)
        except Exception as exc:
            extractor_error = str(exc)

    return {
        "model_id":        model_id,
        "metadata":        metadata,
        "sections":        sections,
        "extractor_error": extractor_error,
    }
