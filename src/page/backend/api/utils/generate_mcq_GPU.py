"""
mcq_generator.py
----------------
Module autonome pour la génération de QCM via Ollama ou HuggingFace.

Fonctions publiques
-------------------
- generate_mcq(full_prompt, model_name, temperature=0.1) -> str
    Génération via Ollama (structured output).

- generate_mcq_hf(content, model_name, tokenizer, temperature=0.1) -> str
    Génération via un modèle HuggingFace local (GPU).

- validate_mcq(mcq_json) -> MCQQuestion | None
    Valide et désérialise un JSON en MCQQuestion.
"""

import ast
import json
from typing import Optional

from pydantic import BaseModel, ValidationError
from ollama import Client
from transformers import AutoTokenizer, pipeline

_ollama = Client(timeout=600)  # 10 min max


# ---------------------------------------------------------------------------
# Schéma de données
# ---------------------------------------------------------------------------

class MCQQuestion(BaseModel):
    question: str
    question_comment: Optional[str] = ""
    option_a: str
    option_a_comment: Optional[str] = ""
    option_b: str
    option_b_comment: Optional[str] = ""
    option_c: str
    option_c_comment: Optional[str] = ""
    option_d: str
    option_d_comment: Optional[str] = ""
    correct_option: str


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def validate_mcq(mcq_json: str) -> Optional[MCQQuestion]:
    """Valide et désérialise un JSON brut en MCQQuestion.
    Retourne None si la validation échoue.
    """
    try:
        return MCQQuestion.model_validate_json(mcq_json)
    except ValidationError as e:
        print(f"Validation failed: {e}")
        return None


def extract_json(text: str) -> str:
    """Extrait le dernier objet JSON valide d'une chaîne de texte (après un éventuel bloc <think>)."""
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Aucun objet JSON trouvé dans la réponse: {text[:200]}")
    text = text[start:end + 1]
    return json.dumps(ast.literal_eval(text), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Génération via Ollama
# ---------------------------------------------------------------------------

def generate_mcq(full_prompt: str, model_name: str, temperature: float = 0.1) -> str:
    """Génère un QCM au format JSON structuré via Ollama.

    Args:
        full_prompt:  Prompt complet envoyé au modèle.
        model_name:   Nom du modèle Ollama (ex. "llama3.1:70b").
        temperature:  Température d'échantillonnage (défaut 0.1).

    Returns:
        Chaîne JSON brute retournée par le modèle.
    """
    generate_params = {
        "model": model_name,
        "options": {"temperature": temperature, "num_ctx": 8192, "top_p": 1},
        "prompt": full_prompt,
    }
    response = _ollama.generate(**generate_params)
    print(f"[generate_mcq] raw response: {response.response}")
    return extract_json(response.response)


# ---------------------------------------------------------------------------
# Génération via HuggingFace (GPU)
# ---------------------------------------------------------------------------

_HF_PROMPT_TEMPLATE = """
À partir du contenu éducatif suivant, générez exactement une question à choix multiple avec quatre options de réponse (a, b, c, d), dont une seule est correcte.

OBJECTIFS :
- La question doit évaluer la compréhension des idées principales.
- Les distracteurs doivent être plausibles mais incorrects.
- Les options doivent être courtes.
- Fournir une justification pédagogique pour chaque option.
- Fournir un commentaire global pour la question.

CONTRAINTES STRICTES DE SORTIE :
1. La sortie doit être STRICTEMENT un unique objet JSON valide.
2. Interdiction ABSOLUE d'ajouter :
   - des blocs ```json
   - du texte avant ou après le JSON
   - des explications hors champs JSON
3. Le champ "correct_option" doit contenir EXACTEMENT une lettre minuscule parmi : "a", "b", "c", "d".
4. Utiliser uniquement des doubles quotes : "..."
5. Le JSON doit contenir EXACTEMENT les 11 champs suivants :

{{
  "question": "...",
  "question_comment": "...",
  "option_a": "...",
  "option_a_comment": "...",
  "option_b": "...",
  "option_b_comment": "...",
  "option_c": "...",
  "option_c_comment": "...",
  "option_d": "...",
  "option_d_comment": "...",
  "correct_option": "a"
}}

RÈGLES POUR LES COMMENTAIRES :
- Chaque commentaire d'option doit expliquer brièvement pourquoi l'option est correcte ou incorrecte.
- Le commentaire global de la question doit expliquer ce que la question évalue ou signaler un piège courant.
- Les commentaires doivent être factuels, concis et pédagogiques.

CONTENU ÉDUCATIF :
{content}

INSTRUCTION FINALE :
Répondez UNIQUEMENT avec un unique objet JSON valide, sans aucun texte en dehors.
"""


def generate_mcq_hf(
    content: str,
    model_name: str,
    tokenizer: AutoTokenizer,
    temperature: float = 0.1,
) -> str:
    """Génère un QCM via un modèle HuggingFace local (requiert un GPU CUDA).

    Args:
        content:      Contenu éducatif source.
        model_name:   Identifiant HuggingFace du modèle (ex. "meta-llama/Llama-3.1-8B-Instruct").
        tokenizer:    Tokenizer déjà chargé pour ce modèle.
        temperature:  Température d'échantillonnage (défaut 0.1).

    Returns:
        Chaîne JSON brute extraite de la réponse du modèle.
    """
    prompt = _HF_PROMPT_TEMPLATE.format(content=content)

    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        device_map="cuda",
        dtype="bfloat16",
    )

    messages = [{"role": "user", "content": prompt}]

    response = pipe(
        messages,
        max_new_tokens=2048,
        temperature=temperature,
        top_p=1.0,
        do_sample=True,
        return_full_text=False,
    )

    return extract_json(response[0]["generated_text"])