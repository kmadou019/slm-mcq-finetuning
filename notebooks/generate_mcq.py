#!/usr/bin/env python3

# # Prerequisite

from sys import argv
from typing import Optional
from pathlib import Path
import os
import re
import unicodedata
import ast
import json

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"

import requests
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, ValidationError
from ollama import generate, Client as OllamaClient
from transformers import AutoTokenizer, pipeline
from dotenv import load_dotenv
from huggingface_hub import login
import ollama
from openai import OpenAI

# ---------------------------------------------------------------------------
# LISA prompt builder (duplicated from src/page/backend/api/utils/prompt_builder.py)
# — this module is intentionally independent from src/page/
# ---------------------------------------------------------------------------

_KEYWORD_MODEL = "hf.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q8_0"


def _cfg():
    return (
        os.getenv("GRAPHDB_MCP_URL", ""),
        os.getenv("GRAPHDB_BEARER_TOKEN", ""),
        os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )


_QCM_GUIDELINES = """\
Règles pour la question (stem) :
- La question doit être compréhensible sans lire les propositions
- Rédiger à la forme affirmative — éviter la négation ("laquelle n'est PAS…")
- Ne pas surcharger le stem d'informations non pertinentes à l'objectif évalué

Règles pour les propositions :
- 4 propositions (a, b, c, d), une seule exacte (QRU)
- Propositions homogènes, parallèles, d'un niveau de granularité similaire
- Propositions exprimées à la forme affirmative
- Longueur et précision similaires entre toutes les propositions — la bonne réponse ne doit pas se distinguer par sa longueur
- La bonne réponse ne doit pas reprendre les mots du stem (cluing)
- Distracteurs plausibles mais factuellement incorrects
- Propositions courtes et concises

Propositions INTERDITES :
- "Toutes les propositions précédentes sont correctes"
- "Aucune des propositions précédentes"
- "A et C sont correctes" (combinaison de propositions)
- Toute proposition absurde ou trivialement éliminable

Justifications :
- Fournir une justification pédagogique pour chaque proposition (correcte ou incorrecte)
- Fournir un commentaire global sur ce que la question évalue ou un piège courant"""

_JSON_OUTPUT_BLOCK = """\
CONTRAINTES STRICTES DE SORTIE :
1. La sortie doit être STRICTEMENT un unique objet JSON valide.
2. Interdiction ABSOLUE d'ajouter :
   - des blocs ```json
   - du texte avant ou après le JSON
   - des explications hors champs JSON
3. Le champ "correct_option" doit contenir EXACTEMENT une lettre minuscule parmi : "a", "b", "c", "d".
4. Utiliser uniquement des doubles quotes : "..."
5. Le JSON doit contenir EXACTEMENT les 11 champs suivants :

{
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
}

RÈGLES POUR LES COMMENTAIRES :
- Chaque commentaire d'option doit expliquer brièvement pourquoi l'option est correcte ou incorrecte.
- Le commentaire global de la question doit expliquer ce que la question évalue ou signaler un piège courant.
- Les commentaires doivent être factuels, concis et pédagogiques."""


def _mcp_headers(session_id=""):
    _, token, _ = _cfg()
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
         "Accept": "application/json, text/event-stream"}
    if session_id:
        h["mcp-session-id"] = session_id
    return h


def _parse_sse_data(text):
    for line in text.splitlines():
        if line.startswith("data: "):
            try:
                return json.loads(line[6:])
            except json.JSONDecodeError:
                pass
    return {}


def _mcp_init_session():
    mcp_url, _, _ = _cfg()
    payload = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
               "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                          "clientInfo": {"name": "prompt-builder", "version": "1.0"}}}
    resp = requests.post(mcp_url, json=payload, headers=_mcp_headers(), timeout=10)
    resp.raise_for_status()
    return resp.headers.get("mcp-session-id", "")


def _mcp_sparql(sparql, session_id):
    mcp_url, _, _ = _cfg()
    payload = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
               "params": {"name": "sparqlQuery", "arguments": {"query": sparql, "format": "json"}}}
    resp = requests.post(mcp_url, json=payload, headers=_mcp_headers(session_id), timeout=15)
    resp.raise_for_status()
    data = _parse_sse_data(resp.text)
    text_content = data.get("result", {}).get("content", [{}])[0].get("text", "{}")
    return json.loads(text_content).get("results", {}).get("bindings", [])


def _search_items(term, session_id):
    escaped = term.replace('"', '\\"').replace("\\", "\\\\")
    sparql = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX lisa: <https://uness.fr/lisa/ontology/>
SELECT ?item ?label WHERE {{
  ?item rdfs:label ?label .
  ?item rdf:type lisa:KnowledgeItem .
  FILTER(regex(?label, "{escaped}", "i"))
}} LIMIT 10
"""
    return _mcp_sparql(sparql, session_id)


def _get_objective_uris(item_uri, session_id):
    sparql = f"""
PREFIX lisa: <https://uness.fr/lisa/ontology/>
SELECT ?objective WHERE {{
  <{item_uri}> lisa:hasKnowledgeObjective ?objective .
}}
"""
    return [b["objective"]["value"] for b in _mcp_sparql(sparql, session_id) if "objective" in b]


def _get_objective_attrs(uri, session_id):
    sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX lisa: <https://uness.fr/lisa/ontology/>
SELECT ?property ?value WHERE {{
  <{uri}> ?property ?value .
}}
"""
    attrs = {}
    for b in _mcp_sparql(sparql, session_id):
        prop = b.get("property", {}).get("value", "")
        val = b.get("value", {}).get("value", "")
        if prop.endswith("#label") or prop.endswith("/label"):
            attrs["label"] = val
        elif prop.endswith("rank"):
            attrs["rank"] = val
        elif prop.endswith("order"):
            try:
                attrs["order"] = int(val)
            except ValueError:
                pass
    return attrs


def _remove_accents(text):
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _fetch_objectives(search_term):
    try:
        session_id = _mcp_init_session()
    except Exception as e:
        print(f"[prompt_builder] MCP session init failed: {e}")
        return None, None, []

    candidates = [search_term]
    first_word = search_term.split()[0] if search_term.split() else ""
    if first_word and first_word != search_term:
        candidates.append(first_word)
    no_accent = _remove_accents(search_term)
    if no_accent != search_term:
        candidates.append(no_accent)
    m = re.search(r"\d+", search_term)
    if m:
        candidates.append(m.group())

    items = []
    for term in candidates:
        try:
            items = _search_items(term, session_id)
        except Exception as e:
            print(f"[prompt_builder] SPARQL search failed for '{term}': {e}")
        if items:
            break

    if not items:
        return None, None, []

    first = items[0]
    item_uri = first["item"]["value"]
    item_label = first.get("label", {}).get("value", "")
    m2 = re.search(r"KnowledgeItem(\d+)$", item_uri)
    item_ref = f"Item {m2.group(1)}" if m2 else item_uri.split("/")[-1]

    try:
        obj_uris = _get_objective_uris(item_uri, session_id)
    except Exception as e:
        print(f"[prompt_builder] Get objectives failed: {e}")
        return item_ref, item_label, []

    objectives = []
    for uri in obj_uris:
        try:
            attrs = _get_objective_attrs(uri, session_id)
        except Exception:
            continue
        if "label" in attrs:
            objectives.append({"label": attrs["label"], "rank": attrs.get("rank", "B"),
                                "order": attrs.get("order", 999)})
    objectives.sort(key=lambda x: x["order"])
    return item_ref, item_label, objectives


def _extract_via_regex(content):
    head = content[:2000]
    m = re.search(r"\|Item_parent\s*=\s*([^\n|]+)", head)
    if m:
        val = re.split(r"[.|]", m.group(1).strip())[0].strip()
        if len(val) > 3:
            return val[:120]
    for line in head.splitlines():
        line = re.sub(r"^#+\s*", "", line.strip())
        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if len(line) < 5:
            continue
        cleaned = re.sub(r"\bitem\s+\d+\b", "", line, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*[-–—]\s*", " ", cleaned).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) > 3:
            return cleaned[:80]
    return None


def _extract_via_ollama(content):
    lines = content.splitlines()
    title_lines = [l.strip() for l in lines if l.strip() and 5 < len(l.strip()) < 150]
    excerpt = "\n".join(title_lines[:30]) if title_lines else content[:3000]
    try:
        _, _, ollama_host = _cfg()
        client = OllamaClient(host=ollama_host, timeout=30)
        prompt = (
            "Tu es un assistant spécialisé dans le curriculum médical français ECNi/LISA.\n"
            "Lis ce contenu pédagogique et identifie les 3 termes de recherche les plus pertinents "
            "pour retrouver l'item LISA correspondant dans la base de données LISA ECNi.\n\n"
            "Les labels LISA sont des phrases nominales comme :\n"
            "- \"Insuffisance cardiaque de l'adulte\"\n"
            "- \"Relation médecin-malade\"\n"
            "- \"Facteurs de risque cardio-vasculaire\"\n\n"
            "Réponds UNIQUEMENT avec 3 termes de recherche, un par ligne, du plus au moins spécifique. "
            "Pas de numérotation, pas d'explication, pas de ponctuation finale.\n\n"
            f"CONTENU :\n{excerpt}"
        )
        response = client.generate(model=_KEYWORD_MODEL, prompt=prompt,
                                   options={"temperature": 0.0, "num_predict": 60})
        raw = response.response.strip()
        return [
            line.strip().strip(".,;:-\"'\n")
            for line in raw.splitlines()
            if line.strip() and len(line.strip()) > 3
        ][:3]
    except Exception as e:
        print(f"[prompt_builder] Ollama keyword extraction failed: {e}")
        return []


def _build_enriched_prompt(item_ref, item_label, objectives):
    obj_lines = [f"- {o['label']} [Rang {o['rank'].upper()}]" for o in objectives]
    objectives_str = "\n".join(obj_lines)
    item_line = item_ref + (f" — {item_label}" if item_label else "")
    rang_a = [o for o in objectives if o.get("rank", "").upper() == "A"]
    rang_a_str = "\n".join(f"- {o['label']} [Rang A]" for o in rang_a) or objectives_str

    return (
        f"### ITEM ET RÉFÉRENCE\n\n{item_line}\n\n"
        f"### OBJECTIFS DE CONNAISSANCE\n\n"
        f"La question doit évaluer l'un des objectifs officiels de cet item.\n"
        f"Priorité absolue aux objectifs de Rang A.\n\n{objectives_str}\n\n"
        f"### OBJECTIF CIBLÉ\n\n"
        f"Sélectionner l'objectif de Rang A le plus pertinent au regard du contenu fourni.\n"
        f"La question doit évaluer exclusivement cet objectif — toute information hors périmètre est à écarter.\n\n"
        f"Objectifs Rang A disponibles :\n{rang_a_str}\n\n"
        f"### CONSIGNES DE RÉDACTION\n\n"
        f"Respecter exclusivement les informations et objectifs présents dans le contenu fourni.\n"
        f"Rédiger en français clair, précis et sans ambiguïté.\n"
        f"La question est indépendante (Question Isolée) — aucun contexte clinique narratif n'est requis.\n\n"
        f"{_QCM_GUIDELINES}\n\n{_JSON_OUTPUT_BLOCK}\n\n"
        f"### CONTENU SOURCE\n\n{{content}}\n\n"
        f"INSTRUCTION FINALE :\nRépondez UNIQUEMENT avec un unique objet JSON valide, sans aucun texte en dehors."
    )


def _build_fallback_prompt():
    return (
        f"À partir du contenu éducatif suivant, générez exactement une question à choix unique "
        f"avec quatre options de réponse (a, b, c, d), dont une seule est correcte.\n\n"
        f"CONSIGNES DE RÉDACTION :\n"
        f"La question doit évaluer la compréhension des idées principales du contenu fourni.\n\n"
        f"{_QCM_GUIDELINES}\n\n{_JSON_OUTPUT_BLOCK}\n\n"
        f"CONTENU ÉDUCATIF :\n{{content}}\n\n"
        f"INSTRUCTION FINALE :\nRépondez UNIQUEMENT avec un unique objet JSON valide, sans aucun texte en dehors."
    )


def _build_prompt(content: str) -> str:
    mcp_url, _, _ = _cfg()
    if not mcp_url:
        return _build_fallback_prompt().replace("{content}", content)

    search_term = _extract_via_regex(content)
    item_ref, item_label, objectives = (None, None, [])
    if search_term:
        item_ref, item_label, objectives = _fetch_objectives(search_term)
    if not objectives:
        for candidate in _extract_via_ollama(content):
            item_ref, item_label, objectives = _fetch_objectives(candidate)
            if objectives:
                break

    if objectives:
        return _build_enriched_prompt(item_ref, item_label, objectives).replace("{content}", content)
    return _build_fallback_prompt().replace("{content}", content)


# ---------------------------------------------------------------------------
# LISA prompt builder
# ---------------------------------------------------------------------------

_KEYWORD_MODEL = "mistral-small3.2"


def _cfg():
    return (
        os.getenv("GRAPHDB_MCP_URL", ""),
        os.getenv("GRAPHDB_BEARER_TOKEN", ""),
        os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )


_QCM_GUIDELINES = """\
Règles pour la question (stem) :
- La question doit être compréhensible sans lire les propositions
- Rédiger à la forme affirmative — éviter la négation ("laquelle n'est PAS…")
- Ne pas surcharger le stem d'informations non pertinentes à l'objectif évalué

Règles pour les propositions :
- 4 propositions (a, b, c, d), une seule exacte (QRU)
- Propositions homogènes, parallèles, d'un niveau de granularité similaire
- Propositions exprimées à la forme affirmative
- Longueur et précision similaires entre toutes les propositions — la bonne réponse ne doit pas se distinguer par sa longueur
- La bonne réponse ne doit pas reprendre les mots du stem (cluing)
- Distracteurs plausibles mais factuellement incorrects
- Propositions courtes et concises

Propositions INTERDITES :
- "Toutes les propositions précédentes sont correctes"
- "Aucune des propositions précédentes"
- "A et C sont correctes" (combinaison de propositions)
- Toute proposition absurde ou trivialement éliminable

Justifications :
- Fournir une justification pédagogique pour chaque proposition (correcte ou incorrecte)
- Fournir un commentaire global sur ce que la question évalue ou un piège courant"""

_JSON_OUTPUT_BLOCK = """\
CONTRAINTES STRICTES DE SORTIE :
1. La sortie doit être STRICTEMENT un unique objet JSON valide.
2. Interdiction ABSOLUE d'ajouter :
   - des blocs ```json
   - du texte avant ou après le JSON
   - des explications hors champs JSON
3. Le champ "correct_option" doit contenir EXACTEMENT une lettre minuscule parmi : "a", "b", "c", "d".
4. Utiliser uniquement des doubles quotes : "..."
5. Le JSON doit contenir EXACTEMENT les 11 champs suivants :

{
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
}

RÈGLES POUR LES COMMENTAIRES :
- Chaque commentaire d'option doit expliquer brièvement pourquoi l'option est correcte ou incorrecte.
- Le commentaire global de la question doit expliquer ce que la question évalue ou signaler un piège courant.
- Les commentaires doivent être factuels, concis et pédagogiques."""


def _mcp_headers(session_id=""):
    _, token, _ = _cfg()
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
         "Accept": "application/json, text/event-stream"}
    if session_id:
        h["mcp-session-id"] = session_id
    return h


def _parse_sse_data(text):
    for line in text.splitlines():
        if line.startswith("data: "):
            try:
                return json.loads(line[6:])
            except json.JSONDecodeError:
                pass
    return {}


def _mcp_init_session():
    mcp_url, _, _ = _cfg()
    payload = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
               "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                          "clientInfo": {"name": "prompt-builder", "version": "1.0"}}}
    resp = requests.post(mcp_url, json=payload, headers=_mcp_headers(), timeout=10, verify=False)
    resp.raise_for_status()
    return resp.headers.get("mcp-session-id", "")


def _mcp_sparql(sparql, session_id):
    mcp_url, _, _ = _cfg()
    payload = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
               "params": {"name": "sparqlQuery", "arguments": {"query": sparql, "format": "json"}}}
    resp = requests.post(mcp_url, json=payload, headers=_mcp_headers(session_id), timeout=15, verify=False)
    resp.raise_for_status()
    data = _parse_sse_data(resp.text)
    text_content = data.get("result", {}).get("content", [{}])[0].get("text", "{}")
    return json.loads(text_content).get("results", {}).get("bindings", [])


def _search_items(term, session_id):
    escaped = term.replace('"', '\\"').replace("\\", "\\\\")
    sparql = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX lisa: <https://uness.fr/lisa/ontology/>
SELECT ?item ?label WHERE {{
  ?item rdfs:label ?label .
  ?item rdf:type lisa:KnowledgeItem .
  FILTER(regex(?label, "{escaped}", "i"))
}} LIMIT 10
"""
    return _mcp_sparql(sparql, session_id)


def _get_objective_uris(item_uri, session_id):
    sparql = f"""
PREFIX lisa: <https://uness.fr/lisa/ontology/>
SELECT ?objective WHERE {{
  <{item_uri}> lisa:hasKnowledgeObjective ?objective .
}}
"""
    return [b["objective"]["value"] for b in _mcp_sparql(sparql, session_id) if "objective" in b]


def _get_objective_attrs(uri, session_id):
    sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX lisa: <https://uness.fr/lisa/ontology/>
SELECT ?property ?value WHERE {{
  <{uri}> ?property ?value .
}}
"""
    attrs = {}
    for b in _mcp_sparql(sparql, session_id):
        prop = b.get("property", {}).get("value", "")
        val = b.get("value", {}).get("value", "")
        if prop.endswith("#label") or prop.endswith("/label"):
            attrs["label"] = val
        elif prop.endswith("rank"):
            attrs["rank"] = val
        elif prop.endswith("order"):
            try:
                attrs["order"] = int(val)
            except ValueError:
                pass
    return attrs


def _remove_accents(text):
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


_objectives_cache = {}


def _fetch_objectives(search_term):
    if search_term in _objectives_cache:
        return _objectives_cache[search_term]

    print(f"[prompt_builder] searching GraphDB for: '{search_term}'")
    try:
        session_id = _mcp_init_session()
    except Exception as e:
        print(f"[prompt_builder] MCP session init failed: {e}")
        return None, None, []

    candidates = [search_term]
    first_word = search_term.split()[0] if search_term.split() else ""
    if first_word and first_word != search_term:
        candidates.append(first_word)
    no_accent = _remove_accents(search_term)
    if no_accent != search_term:
        candidates.append(no_accent)
    m = re.search(r"\d+", search_term)
    if m:
        candidates.append(m.group())

    items = []
    for term in candidates:
        try:
            items = _search_items(term, session_id)
        except Exception as e:
            print(f"[prompt_builder] SPARQL search failed for '{term}': {e}")
        if items:
            break

    if not items:
        return None, None, []

    first = items[0]
    item_uri = first["item"]["value"]
    item_label = first.get("label", {}).get("value", "")
    m2 = re.search(r"KnowledgeItem(\d+)$", item_uri)
    item_ref = f"Item {m2.group(1)}" if m2 else item_uri.split("/")[-1]

    try:
        obj_uris = _get_objective_uris(item_uri, session_id)
    except Exception as e:
        print(f"[prompt_builder] Get objectives failed: {e}")
        return item_ref, item_label, []

    objectives = []
    for uri in obj_uris:
        try:
            attrs = _get_objective_attrs(uri, session_id)
        except Exception:
            continue
        if "label" in attrs:
            objectives.append({"label": attrs["label"], "rank": attrs.get("rank", "B"),
                                "order": attrs.get("order", 999)})
    objectives.sort(key=lambda x: x["order"])
    result = item_ref, item_label, objectives
    _objectives_cache[search_term] = result
    return result


def _extract_via_regex(content):
    head = content[:2000]
    m = re.search(r"\|Item_parent\s*=\s*([^\n|]+)", head)
    if m:
        val = re.split(r"[.|]", m.group(1).strip())[0].strip()
        if len(val) > 3:
            return val[:120]
    for line in head.splitlines():
        line = re.sub(r"^#+\s*", "", line.strip())
        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if len(line) < 5:
            continue
        cleaned = re.sub(r"\bitem\s+\d+\b", "", line, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*[-–—]\s*", " ", cleaned).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) > 3:
            return cleaned[:80]
    return None


def _extract_via_ollama(content):
    lines = content.splitlines()
    title_lines = [l.strip() for l in lines if l.strip() and 5 < len(l.strip()) < 150]
    excerpt = "\n".join(title_lines[:30]) if title_lines else content[:3000]
    try:
        _, _, ollama_host = _cfg()
        client = OllamaClient(host=ollama_host, timeout=30)
        prompt = (
            "Tu es un assistant spécialisé dans le curriculum médical français ECNi/LISA.\n"
            "Lis ce contenu pédagogique et identifie les 3 termes de recherche les plus pertinents "
            "pour retrouver l'item LISA correspondant dans la base de données LISA ECNi.\n\n"
            "Les labels LISA sont des phrases nominales comme :\n"
            "- \"Insuffisance cardiaque de l'adulte\"\n"
            "- \"Relation médecin-malade\"\n"
            "- \"Facteurs de risque cardio-vasculaire\"\n\n"
            "Réponds UNIQUEMENT avec 3 termes de recherche, un par ligne, du plus au moins spécifique. "
            "Pas de numérotation, pas d'explication, pas de ponctuation finale.\n\n"
            f"CONTENU :\n{excerpt}"
        )
        response = client.generate(model=_KEYWORD_MODEL, prompt=prompt,
                                   options={"temperature": 0.0, "num_predict": 60})
        raw = response.response.strip()
        return [
            line.strip().strip(".,;:-\"'\n")
            for line in raw.splitlines()
            if line.strip() and len(line.strip()) > 3
        ][:3]
    except Exception as e:
        print(f"[prompt_builder] Ollama keyword extraction failed: {e}")
        return []


def _build_enriched_prompt(item_ref, item_label, objectives):
    obj_lines = [f"- {o['label']} [Rang {o['rank'].upper()}]" for o in objectives]
    objectives_str = "\n".join(obj_lines)
    item_line = item_ref + (f" — {item_label}" if item_label else "")
    rang_a = [o for o in objectives if o.get("rank", "").upper() == "A"]
    rang_a_str = "\n".join(f"- {o['label']} [Rang A]" for o in rang_a) or objectives_str

    return (
        f"### ITEM ET RÉFÉRENCE\n\n{item_line}\n\n"
        f"### OBJECTIFS DE CONNAISSANCE\n\n"
        f"La question doit évaluer l'un des objectifs officiels de cet item.\n"
        f"Priorité absolue aux objectifs de Rang A.\n\n{objectives_str}\n\n"
        f"### OBJECTIF CIBLÉ\n\n"
        f"Sélectionner l'objectif de Rang A le plus pertinent au regard du contenu fourni.\n"
        f"La question doit évaluer exclusivement cet objectif — toute information hors périmètre est à écarter.\n\n"
        f"Objectifs Rang A disponibles :\n{rang_a_str}\n\n"
        f"### CONSIGNES DE RÉDACTION\n\n"
        f"Respecter exclusivement les informations et objectifs présents dans le contenu fourni.\n"
        f"Rédiger en français clair, précis et sans ambiguïté.\n"
        f"La question est indépendante (Question Isolée) — aucun contexte clinique narratif n'est requis.\n\n"
        f"{_QCM_GUIDELINES}\n\n{_JSON_OUTPUT_BLOCK}\n\n"
        f"### CONTENU SOURCE\n\n{{content}}\n\n"
        f"INSTRUCTION FINALE :\nRépondez UNIQUEMENT avec un unique objet JSON valide, sans aucun texte en dehors."
    )


def _build_fallback_prompt():
    return (
        f"À partir du contenu éducatif suivant, générez exactement une question à choix unique "
        f"avec quatre options de réponse (a, b, c, d), dont une seule est correcte.\n\n"
        f"CONSIGNES DE RÉDACTION :\n"
        f"La question doit évaluer la compréhension des idées principales du contenu fourni.\n\n"
        f"{_QCM_GUIDELINES}\n\n{_JSON_OUTPUT_BLOCK}\n\n"
        f"CONTENU ÉDUCATIF :\n{{content}}\n\n"
        f"INSTRUCTION FINALE :\nRépondez UNIQUEMENT avec un unique objet JSON valide, sans aucun texte en dehors."
    )


def _build_old_prompt_ollama(content):
    """Ancien système de prompt (commit ba23972) — utilisé via Ollama."""
    return f"""
        À partir du contenu éducatif suivant, générez une question à choix multiple avec quatre options de réponse dont une seule est correcte.
        La question doit évaluer la compréhension des idées principales, et les options doivent être claires, informatives et pertinentes.
        Assurez-vous que les distracteurs (options incorrectes) suivent une interprétation logique mais incorrecte, basée sur des idées reçues ou des incompréhensions courantes du sujet.
        Les options de réponse doivent être aussi courtes que possible.

        IMPORTANT — FORMAT ABSOLU POUR LE CHAMP 'correct_option' :
        - Ce champ doit contenir exactement **une seule lettre minuscule** parmi : a, b, c ou d.
        - **Exemples valides** : "a", "b", "c", "d".
        - **Interdits** : "a)", "A", "a.", "a )", "le texte de la réponse correcte", 1, true, etc.
        - La sortie JSON doit conserver ce champ comme chaîne (`"correct_option": "a"`).

        Fournissez la sortie strictement au format JSON correspondant au schéma demandé (ne pas produire de texte hors-du-JSON).
        **Contenu éducatif :**
        {content}
    """


def _build_old_prompt_hf(content):
    """Ancien système de prompt (commit ba23972) — utilisé via HuggingFace."""
    return f"""
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
    """


_SENTINEL = "<<<CONTENU_EDUCATIF>>>"
_QWEN_OPTIMIZED_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "optimized_prompts", "qwen3_0_6b", "final_prompt.txt"
)


def _load_optimized_prompt(save_name: str) -> str | None:
    project_root = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(project_root, "data", "optimized_prompts", save_name, "final_prompt.txt")
    if not os.path.exists(path):
        print(f"[prompt_builder] prompt optimisé introuvable : {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def _build_prompt(content, hf=False):
    """
    Sélectionne le système de prompt via la variable d'env PROMPT_SYSTEM :
      - "old"                : ancien prompt simple (commit ba23972)
      - "new" (défaut)       : prompt enrichi via retrieval GraphDB, fallback générique si indisponible
      - "optimized"          : prompt optimisé du modèle courant (OPTIMIZED_PROMPT_SAVE_NAME)
      - "optimized_enriched" : idem + objectifs LISA injectés dans le contenu
      - "transfer_qwen"      : prompt optimisé de qwen3_0_6b appliqué à n'importe quel modèle
    """
    prompt_system = os.getenv("PROMPT_SYSTEM", "new").lower()

    if prompt_system == "old":
        print("[prompt_builder] PROMPT_SYSTEM=old → ancien prompt")
        return _build_old_prompt_hf(content) if hf else _build_old_prompt_ollama(content)

    if prompt_system in ("optimized", "optimized_enriched", "transfer_qwen"):
        if prompt_system == "transfer_qwen":
            save_name = "qwen3_0_6b"
        else:
            save_name = os.getenv("OPTIMIZED_PROMPT_SAVE_NAME", "")
            if not save_name:
                print("[prompt_builder] OPTIMIZED_PROMPT_SAVE_NAME non défini → fallback prompt")
                return _build_fallback_prompt().replace("{content}", content)

        template = _load_optimized_prompt(save_name)
        if template is None or _SENTINEL not in template:
            print(f"[prompt_builder] prompt optimisé invalide pour {save_name} → fallback")
            return _build_fallback_prompt().replace("{content}", content)

        if prompt_system == "optimized_enriched":
            # Enrichir le contenu avec les objectifs LISA avant injection
            enriched_content = _build_lisa_enrichment(content)
            print(f"[prompt_builder] {prompt_system} → {save_name} + enrichissement LISA")
            return template.replace(_SENTINEL, enriched_content)

        print(f"[prompt_builder] {prompt_system} → prompt optimisé {save_name}")
        return template.replace(_SENTINEL, content)

    # Système nouveau (retrieval GraphDB)
    mcp_url, _, _ = _cfg()
    if not mcp_url:
        print("[prompt_builder] GRAPHDB_MCP_URL not set → fallback prompt")
        return _build_fallback_prompt().replace("{content}", content)

    search_term = _extract_via_regex(content)
    item_ref, item_label, objectives = (None, None, [])
    if search_term:
        item_ref, item_label, objectives = _fetch_objectives(search_term)
    if not objectives:
        for candidate in _extract_via_ollama(content):
            item_ref, item_label, objectives = _fetch_objectives(candidate)
            if objectives:
                break

    if objectives:
        print(f"[prompt_builder] enriched prompt → {item_ref} ({len(objectives)} objectives)")
        return _build_enriched_prompt(item_ref, item_label, objectives).replace("{content}", content)
    print("[prompt_builder] no objectives found → fallback prompt")
    return _build_fallback_prompt().replace("{content}", content)


def _build_lisa_enrichment(content: str) -> str:
    """
    Construit un contenu enrichi : objectifs LISA officiels préfixés au contenu brut.
    Fallback vers le contenu brut si GraphDB/Ollama indisponible.
    """
    search_term = _extract_via_regex(content)
    item_ref, item_label, objectives = (None, None, [])
    if search_term:
        item_ref, item_label, objectives = _fetch_objectives(search_term)
    if not objectives:
        for candidate in _extract_via_ollama(content):
            item_ref, item_label, objectives = _fetch_objectives(candidate)
            if objectives:
                break

    if not objectives:
        print("[prompt_builder] enrichissement LISA : aucun objectif trouvé → contenu brut")
        return content

    obj_lines = "\n".join(f"- {o['label']} [Rang {o['rank'].upper()}]" for o in objectives)
    item_line = item_ref + (f" — {item_label}" if item_label else "")
    prefix = (
        f"### ITEM ET RÉFÉRENCE\n{item_line}\n\n"
        f"### OBJECTIFS DE CONNAISSANCE\n{obj_lines}\n\n"
        f"### CONTENU SOURCE\n"
    )
    print(f"[prompt_builder] enrichissement LISA → {item_ref} ({len(objectives)} objectifs)")
    return prefix + content


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


def validate_mcq(mcq_json):
    try:
        return MCQQuestion.model_validate_json(mcq_json)
    except ValidationError as e:
        print(f"Validation failed: {e}")
        return None


def flatten(df: DataFrame, mcq_column_name: str):
    ids, questions, option_as, option_bs, option_cs, option_ds, correct_options = [], [], [], [], [], [], []
    q_comments = []
    a_comments, b_comments, c_comments, d_comments = [], [], [], []

    for idx, row in df.iterrows():
        mcq = row[mcq_column_name]

        ids.append(row["id"])
        has_mcq = hasattr(mcq, 'question')
        questions.append(mcq.question if has_mcq else "")
        q_comments.append(mcq.question_comment if has_mcq else "")
        option_as.append(mcq.option_a if has_mcq else "")
        a_comments.append(mcq.option_a_comment if has_mcq else "")
        option_bs.append(mcq.option_b if has_mcq else "")
        b_comments.append(mcq.option_b_comment if has_mcq else "")
        option_cs.append(mcq.option_c if has_mcq else "")
        c_comments.append(mcq.option_c_comment if has_mcq else "")
        option_ds.append(mcq.option_d if has_mcq else "")
        d_comments.append(mcq.option_d_comment if has_mcq else "")
        correct_options.append(mcq.correct_option if has_mcq else "")

    return pd.DataFrame({
        "id": ids,
        "question": questions,
        "question_comment": q_comments,
        "option_a": option_as,
        "option_a_comment": a_comments,
        "option_b": option_bs,
        "option_b_comment": b_comments,
        "option_c": option_cs,
        "option_c_comment": c_comments,
        "option_d": option_ds,
        "option_d_comment": d_comments,
        "correct_option": correct_options,
    })


def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    text = text[start:end+1]
    return json.dumps(ast.literal_eval(text), ensure_ascii=False)


def generate_mcq(content, model_name, temperature):
    full_prompt = _build_prompt(content)		
    generate_params = {
        'model': model_name,
        'options': {'temperature': temperature, 'num_ctx': 8192, 'top_p': 1},
        'prompt': full_prompt,
        'format': MCQQuestion.model_json_schema()
    }

    response = generate(**generate_params)
    return response['response']


def generate_mcq_hf(content, pipe, temperature, disable_thinking=False):
    full_prompt = _build_prompt(content, hf=True)

    messages = [{"role": "user", "content": full_prompt}]

    kwargs = dict(
        max_new_tokens=2048,
        temperature=temperature,
        top_p=1.0,
        do_sample=True,
        return_full_text=False,
    )
    if disable_thinking:
        kwargs["tokenize_kwargs"] = {"enable_thinking": False}

    response = pipe(messages, **kwargs)

    raw = response[0]['generated_text']
# Strip any <think>...</think> blocks before JSON extraction
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    return extract_json(raw)


def get_checkpoint(model_name):
    start_path = _DATA / f"checkpoints/start_{model_name}"
    df_path = _DATA / f"checkpoints/df_in_construction_{model_name}.csv"
    try:
        with open(start_path, "r") as start:
            start = start.readline()
            df_in_construction = pd.read_csv(df_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_in_construction = pd.DataFrame()
        start = 0
    return int(start), df_in_construction


def save_checkpoint(start, df_in_construction, model_name):
    os.makedirs(_DATA / "checkpoints", exist_ok=True)
    with open(_DATA / f"checkpoints/start_{model_name}", "w") as fic:
        fic.write(str(start))
    df_in_construction.to_csv(_DATA / f"checkpoints/df_in_construction_{model_name}.csv", index=False)


_THINKING_MODELS = {"nemotron_nano_30b", "qwen3_5_35b_a3b"}


def for_a_model(df_test, model_name, save_name, use_ollama=False):
    disable_thinking = save_name in _THINKING_MODELS
    if not use_ollama:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True
        )
        try:
            pipe = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=tokenizer,
                device_map="cuda",
                dtype="bfloat16"
            )
        except (ValueError, OSError):
            from transformers import AutoModelForCausalLM
            model_obj = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="cuda", torch_dtype="auto", trust_remote_code=True
            )
            pipe = pipeline("text-generation", model=model_obj, tokenizer=tokenizer)
    else:
        pipe = None

    start, df_in_construction = get_checkpoint(save_name)
    pas = 400

    for idx in range(start, len(df_test)):
        content = df_test.loc[idx, "content_raw"]
        nb_try = 0
        while True:
            try:
                generated = (
                    generate_mcq_hf(content, pipe, temperature=0.1, disable_thinking=disable_thinking)
                    if not use_ollama
                    else generate_mcq(content, model_name, temperature=0.1)
                )
                df_in_construction.loc[idx, f"generated_{save_name}"] = generated
                break
            except (ValueError, SyntaxError, KeyError, IndexError) as e:
                print(f"Erreur détectée ({type(e).__name__}), relance...")
                nb_try += 1
                if nb_try == 5:
                    print("Nombre d'essais dépassé, passage au Lisa Sheet suivant")
                    break

        if idx % pas == 0:
            save_checkpoint(idx, df_in_construction, save_name)

    col = f'generated_{save_name}'
    if col not in df_in_construction.columns:
        print(f"[WARN] Aucune génération réussie pour {save_name}, CSV vide.")
        return pd.DataFrame()
    df_test[save_name] = df_in_construction[col].apply(validate_mcq)
    df = flatten(df_test, save_name)

    # Clean checkpoint for this model
    for f in [_DATA / f"checkpoints/df_in_construction_{save_name}.csv",
              _DATA / f"checkpoints/start_{save_name}"]:
        if os.path.exists(f):
            os.remove(f)

    return df


def create_mcq_text(mcq_dict):
    return (
        f"Question: {mcq_dict['question']}\n"
        f"a) {mcq_dict['option_a']}\n"
        f"b) {mcq_dict['option_b']}\n"
        f"c) {mcq_dict['option_c']}\n"
        f"d) {mcq_dict['option_d']}"
    )


def llama_answer_qcm(mcq_text, system_prompt):
    user_prompt = f"""Répond STRICTEMENT à ce QCM :
        {mcq_text}
        CONTRAINTE ABSOLUE :
        - Ta sortie doit être UNIQUEMENT la lettre de la bonne réponse (A, B, C, D, etc.).
        - AUCUN autre texte, aucune explication, aucun point, aucun saut de ligne, aucun espace.
        - Ne préfixe pas la réponse, n'ajoute rien avant ou après.
        - Répond par une seule lettre et rien d'autre.

        FORMAT DE SORTIE OBLIGATOIRE :
        <lettre>
    """
    response = ollama.generate(
        model="llama3.1:70b",
        prompt=user_prompt,
        system=system_prompt)

    return response["response"][0]


def call_openai_api(client, system_prompt, mcq_text, temp=0.5, max_completion_tokens=1):
    user_prompt = f"""Répond STRICTEMENT à ce QCM :
        {mcq_text}
        CONTRAINTE ABSOLUE :
        - Ta sortie doit être UNIQUEMENT la lettre de la bonne réponse (A, B, C, D, etc.).
        - AUCUN autre texte, aucune explication, aucun point, aucun saut de ligne, aucun espace.
        - Ne préfixe pas la réponse, n'ajoute rien avant ou après.
        - Répond par une seule lettre et rien d'autre.

        FORMAT DE SORTIE OBLIGATOIRE :
        <lettre>
         """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=temp,
            max_tokens=max_completion_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


if __name__ == "__main__":
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
    login(token=HF_TOKEN)

    df = pd.read_csv(_DATA / "lisa_sheets.csv")

    with open(_DATA / "train_test_split/test_folders.json", "r", encoding="utf-8") as file:
        test_folders = json.load(file)

    df_test = df[df.folder.isin(test_folders)].reset_index(drop=True)
    print("Number of lisa sheets :", len(df_test))

    # # How many output are incorrect

    def correct_output(df, save_name, file):
        initial_len = len(df)
        df = df[df["correct_option"].astype(str).str.lower().isin(list("abcd"))]

        incorrect_output = initial_len - len(df)
        print(initial_len - incorrect_output)
        print(f"Incorrect output for {save_name}: {round((incorrect_output/initial_len)*100,2)}%", file=file)
        return df

    # ## Correctness

    system_prompt = "Tu es un expert dans le domaine médical"
    client = OpenAI(api_key=OPENAI_KEY)

    def correctness(df, save_name, file):
        initial_len = len(df)
        indices_to_drop = []
        for idx, row in df.iterrows():
            mcq_text = create_mcq_text(row)
            correct_option = row["correct_option"]
            correct_reeval = call_openai_api(client=client, system_prompt=system_prompt, mcq_text=mcq_text)
            if not isinstance(correct_option, str):
                continue
            if correct_reeval is None:
                continue
            if correct_option.lower() != correct_reeval.lower():
                indices_to_drop.append(idx)

        df = df.drop(indices_to_drop).reset_index(drop=True)
        df.to_csv(_DATA / "correct_mcqs_dataset" / f"{save_name}.csv")
        print(f"Number of correct MCQs for {save_name} {len(df)} / {initial_len}", file=file)
        return df

    # # MCQs Generation

    with open(_DATA / "models.json", encoding="utf-8") as _f:
        models = json.load(_f)

    if len(argv) < 2 or argv[1] not in models:
        print("Usage: generate_mcq.py <save_name> [index1] [index2] [start-end] ...")
        print("Valid save names:", list(models.keys()))
        exit(1)

    save_name = argv[1]
    _raw_path = models[save_name]
    # Resolve relative local paths via PROJECT_ROOT env var (set by scripts/env.sh).
    # HuggingFace IDs (e.g. "google/medgemma-4b-it") are left untouched.
    if not os.path.isabs(_raw_path):
        _project_root = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(__file__), ".."))
        _candidate = os.path.join(_project_root, _raw_path)
        if os.path.isdir(_candidate):
            _raw_path = _candidate
    model_name = _raw_path

    # Parsing des index (même logique que main.py)
    indexes = []
    if len(argv) >= 3:
        for arg in argv[2:]:
            if '-' in arg:
                parts = arg.split('-')
                indexes.extend(range(int(parts[0]), int(parts[1]) + 1))
            else:
                indexes.append(int(arg))

    if indexes:
        df_test = df_test.iloc[indexes].reset_index(drop=True)
        print(f"✓ Subset : {len(df_test)} lisa sheets (indexes: {argv[2:]})")

    with open(_ROOT / "correctness.output", mode="a") as f:
        df = for_a_model(df_test, model_name, save_name)
        if df.empty:
            print(f"[WARN] Aucun MCQ généré pour {save_name}, on arrête.")
            exit(1)
        df = correct_output(df, save_name, f)
        df = correctness(df, save_name, f)
