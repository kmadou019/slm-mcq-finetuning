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
import os
import random
from typing import Optional

from pydantic import BaseModel, ValidationError
from ollama import Client
from transformers import AutoTokenizer, pipeline

_ollama = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"), timeout=600)

# OpenAI-compatible endpoint for remote GPU models
_OPENAI_BASE_URL = os.getenv("OPENAI_COMPATIBLE_URL", "http://localhost:4000/v1")
_OPENAI_API_KEY = os.getenv("OPENAI_COMPATIBLE_KEY", "")


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

def shuffle_distractors(mcq: MCQQuestion) -> MCQQuestion:
    """Shuffles all 4 option positions randomly, updating correct_option accordingly.

    Neutralises position bias introduced by the generation model (which often
    places the correct answer in position 'a') and by GPT-4o as evaluator
    (which tends to over-score options presented first).
    """
    letters = ['a', 'b', 'c', 'd']
    options = [
        (mcq.option_a, mcq.option_a_comment),
        (mcq.option_b, mcq.option_b_comment),
        (mcq.option_c, mcq.option_c_comment),
        (mcq.option_d, mcq.option_d_comment),
    ]
    try:
        correct_idx = letters.index(mcq.correct_option.lower())
    except ValueError:
        return mcq  # invalid correct_option (e.g. 'e') — return unchanged

    indices = list(range(4))
    random.shuffle(indices)
    shuffled = [options[i] for i in indices]
    new_correct_idx = indices.index(correct_idx)

    return MCQQuestion(
        question=mcq.question,
        question_comment=mcq.question_comment,
        option_a=shuffled[0][0], option_a_comment=shuffled[0][1],
        option_b=shuffled[1][0], option_b_comment=shuffled[1][1],
        option_c=shuffled[2][0], option_c_comment=shuffled[2][1],
        option_d=shuffled[3][0], option_d_comment=shuffled[3][1],
        correct_option=letters[new_correct_idx],
    )


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
    """Extrait le dernier objet JSON valide d'une chaîne de texte (après un éventuel bloc <think>).
    Si </think> n'est pas fermé (génération tronquée par num_predict), cherche dans le texte entier.
    """
    search_text = text
    if "</think>" in text:
        search_text = text[text.rfind("</think>") + len("</think>"):]

    start = search_text.find("{")
    end   = search_text.rfind("}")
    # Fallback: think block unclosed — search whole text
    if start == -1 or end == -1 or end < start:
        start = text.find("{")
        end   = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Aucun objet JSON trouvé dans la réponse: {text[:200]}")
    text = search_text[start:end + 1]
    try:
        return json.dumps(json.loads(text), ensure_ascii=False)
    except json.JSONDecodeError:
        return json.dumps(ast.literal_eval(text), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Génération via Ollama
# ---------------------------------------------------------------------------

def generate_mcq(full_prompt: str, model_name: str, temperature: float = 0.1) -> str:
    """Génère un QCM au format JSON structuré.

    Route vers Ollama pour les modèles locaux (ollama/*),
    et vers l'endpoint OpenAI-compatible pour les modèles distants (openai/*).

    Args:
        full_prompt:  Prompt complet envoyé au modèle.
        model_name:   Nom du modèle (ex. "qwen3:4b" ou "openai/Qwen/Qwen3.6-35B-A3B-FP8").
        temperature:  Température d'échantillonnage (défaut 0.1).

    Returns:
        Chaîne JSON brute retournée par le modèle.
    """
    # Route based on model prefix
    if model_name.startswith("openai/"):
        # Use OpenAI-compatible endpoint (port 4000 via host.docker.internal)
        return _generate_openai_compat(full_prompt, model_name, temperature)
    else:
        # Use Ollama for local models
        generate_params = {
            "model": model_name,
            "options": {"temperature": temperature, "num_ctx": 8192, "top_p": 1},
            "prompt": full_prompt,
        }
        response = _ollama.generate(**generate_params)
        print(f"[generate_mcq/ollama] raw response: {response.response}")
        return extract_json(response.response)


def _generate_openai_compat(full_prompt: str, model_name: str, temperature: float = 0.1) -> str:
    """Generate MCQ via OpenAI-compatible endpoint (port 4000)."""
    import requests as _requests
    
    headers = {
        "Content-Type": "application/json",
    }
    if _OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {_OPENAI_API_KEY}"
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": temperature,
        "max_tokens": 2048,
    }
    
    print(f"[generate_mcq/openai] calling {model_name} at {_OPENAI_BASE_URL}")
    response = _requests.post(
        f"{_OPENAI_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    
    content = data["choices"][0]["message"]["content"]
    print(f"[generate_mcq/openai] raw response: {content[:500]}...")
    return extract_json(content)


def generate_mcq_chat(full_prompt: str, model_name: str, think: bool = False,
                      temperature: float = 0.1, num_predict: int = 8192) -> str:
    """Génère un QCM via Ollama /api/chat avec contrôle du mode thinking.

    Utiliser cette fonction pour les modèles qui supportent think:true/false
    (Qwen3, Magistral, Nemotron, QwQ, DeepSeek-R1, Phi-4-Reasoning...).

    Args:
        full_prompt:  Prompt complet.
        model_name:   Nom du modèle Ollama.
        think:        True  → active le raisonnement interne.
                      False → désactive le raisonnement (réponse directe).
        temperature:  Température d'échantillonnage (défaut 0.1).
        num_predict:  Nombre max de tokens à générer (thinking trace + JSON).

    Returns:
        Chaîne JSON brute extraite de message.content.
    """
    import re
    response = _ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": full_prompt}],
        think=think,
        options={"temperature": temperature, "num_ctx": 8192, "top_p": 1, "num_predict": num_predict},
    )
    content = response.message.content

    # Print thinking trace — two sources depending on model:
    # 1. Ollama think flag supported (Qwen3, Magistral): trace in message.thinking
    # 2. Inline <think> tags (Nemotron, Phi4-R, QwQ, DeepSeek-R1): trace in content
    thinking = getattr(response.message, "thinking", None)
    if thinking:
        print(f"[THINKING TRACE]\n{thinking}\n[/THINKING TRACE]")
    else:
        m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if m:
            print(f"[THINKING TRACE]\n{m.group(1).strip()}\n[/THINKING TRACE]")

    return extract_json(content)


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