"""
prompt_builder.py
-----------------
Builds an enriched Ollama prompt from uploaded educational content.

Workflow:
  1. Deduce LISA search term via regex (Option A), then Ollama fallback (Option B)
  2. Query GraphDB via MCP protocol (JSON-RPC + SSE) for all knowledge objectives
  3. Assemble an enriched prompt keeping the existing JSON output block intact

The returned prompt always contains a {content} placeholder, compatible with
the existing backend replace logic in _run_generation().
"""

import json
import os
import re
import unicodedata
from typing import Optional

import requests
from ollama import Client as OllamaClient

# ---------------------------------------------------------------------------
# Configuration (from environment — read at call time, not import time)
# ---------------------------------------------------------------------------

_KEYWORD_MODEL: str = "hf.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q8_0"


def _cfg() -> tuple[str, str, str]:
    """Return (mcp_url, token, ollama_host) — re-read from env on each call."""
    return (
        os.getenv("GRAPHDB_MCP_URL", ""),
        os.getenv("GRAPHDB_BEARER_TOKEN", ""),
        os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

# ---------------------------------------------------------------------------
# QCM quality guidelines — shared by enriched and fallback prompts
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# JSON output block — must stay identical to DEFAULT_PROMPT in the frontend
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# MCP protocol helpers
# ---------------------------------------------------------------------------


def _mcp_headers(session_id: str = "") -> dict:
    _, token, _ = _cfg()
    h = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id:
        h["mcp-session-id"] = session_id
    return h


def _parse_sse_data(text: str) -> dict:
    """Extract the JSON payload from an SSE event stream response."""
    for line in text.splitlines():
        if line.startswith("data: "):
            try:
                return json.loads(line[6:])
            except json.JSONDecodeError:
                pass
    return {}


def _mcp_init_session() -> str:
    """Initialize an MCP session and return the session_id."""
    mcp_url, _, _ = _cfg()
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "prompt-builder", "version": "1.0"},
        },
    }
    resp = requests.post(
        mcp_url,
        json=payload,
        headers=_mcp_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.headers.get("mcp-session-id", "")


def _mcp_sparql(sparql: str, session_id: str) -> list[dict]:
    """Execute a SPARQL SELECT query via MCP tools/call. Returns bindings list."""
    mcp_url, _, _ = _cfg()
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "sparqlQuery",
            "arguments": {"query": sparql, "format": "json"},
        },
    }
    resp = requests.post(
        mcp_url,
        json=payload,
        headers=_mcp_headers(session_id),
        timeout=15,
    )
    resp.raise_for_status()
    data = _parse_sse_data(resp.text)
    text_content = (
        data.get("result", {})
        .get("content", [{}])[0]
        .get("text", "{}")
    )
    sparql_result = json.loads(text_content)
    return sparql_result.get("results", {}).get("bindings", [])


# ---------------------------------------------------------------------------
# SPARQL queries (3-hop strategy from SKILL.md)
# ---------------------------------------------------------------------------


def _search_items(term: str, session_id: str) -> list[dict]:
    """Query 1 (Hop 0): search KnowledgeItems matching term."""
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


def _get_objective_uris(item_uri: str, session_id: str) -> list[str]:
    """Query 2 (Hop 1): get objective URIs for an item."""
    sparql = f"""
PREFIX lisa: <https://uness.fr/lisa/ontology/>
SELECT ?objective WHERE {{
  <{item_uri}> lisa:hasKnowledgeObjective ?objective .
}}
"""
    bindings = _mcp_sparql(sparql, session_id)
    return [b["objective"]["value"] for b in bindings if "objective" in b]


def _get_objective_attrs(uri: str, session_id: str) -> dict:
    """Query 3 (Hop 2): get attributes for one objective URI."""
    sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX lisa: <https://uness.fr/lisa/ontology/>
SELECT ?property ?value WHERE {{
  <{uri}> ?property ?value .
}}
"""
    attrs: dict = {}
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


def _remove_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _fetch_objectives(search_term: str) -> tuple[Optional[str], Optional[str], list[dict]]:
    """
    Initialize one MCP session, run 3-hop SPARQL with fallback variants.
    Returns (item_ref, item_label, sorted_objectives).
    """
    try:
        session_id = _mcp_init_session()
    except Exception as e:
        print(f"[prompt_builder] MCP session init failed: {e}")
        return None, None, []

    # Fallback variants: full → first word → no accents → item number
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

    items: list[dict] = []
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

    objectives: list[dict] = []
    for uri in obj_uris:
        try:
            attrs = _get_objective_attrs(uri, session_id)
        except Exception:
            continue
        if "label" in attrs:
            objectives.append({
                "label": attrs["label"],
                "rank": attrs.get("rank", "B"),
                "order": attrs.get("order", 999),
            })
    objectives.sort(key=lambda x: x["order"])

    return item_ref, item_label, objectives


# ---------------------------------------------------------------------------
# Search term extraction
# ---------------------------------------------------------------------------


def _extract_via_regex(content: str) -> Optional[str]:
    """Option A: extract search term from content.

    Priority 1 — wiki field |Item_parent=... (e.g. objectifs de connaissance format).
    Priority 2 — first meaningful line, stripping any 'Item \\d+' marker.
    """
    head = content[:2000]

    # Priority 1: wiki-style |Item_parent=<title>
    m = re.search(r"\|Item_parent\s*=\s*([^\n|]+)", head)
    if m:
        val = m.group(1).strip()
        # Keep only first sentence (up to first period or pipe)
        val = re.split(r"[.|]", val)[0].strip()
        if len(val) > 3:
            return val[:120]

    # Priority 2: first meaningful line (title-based)
    for line in head.splitlines():
        line = re.sub(r"^#+\s*", "", line.strip())         # markdown headers
        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()  # numbered lists
        if len(line) < 5:
            continue

        cleaned = re.sub(r"\bitem\s+\d+\b", "", line, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*[-–—]\s*", " ", cleaned).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if len(cleaned) > 3:
            return cleaned[:80]

    return None


def _extract_via_ollama(content: str) -> Optional[str]:
    """Option B: ask Ollama (plain text) for the main medical keyword."""
    try:
        _, _, ollama_host = _cfg()
        client = OllamaClient(host=ollama_host, timeout=15)
        prompt = (
            "Extrait le mot-clé médical principal de ce contenu pédagogique, "
            "adapté à une recherche dans la base LISA ECNi. "
            "Réponds UNIQUEMENT avec un seul mot ou groupe nominal court, "
            "sans ponctuation finale ni explication.\n\n"
            f"CONTENU :\n{content[:800]}"
        )
        response = client.generate(
            model=_KEYWORD_MODEL,
            prompt=prompt,
            options={"temperature": 0.0, "num_predict": 20},
        )
        keyword = response.response.strip().strip(".,;:\"'\n")
        return keyword or None
    except Exception as e:
        print(f"[prompt_builder] Ollama keyword extraction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _build_enriched_prompt(item_ref: str, item_label: str, objectives: list[dict]) -> str:
    obj_lines = [
        f"- {o['label']} [Rang {o['rank'].upper()}]"
        for o in objectives
    ]
    objectives_str = "\n".join(obj_lines)
    item_line = item_ref + (f" — {item_label}" if item_label else "")

    rang_a = [o for o in objectives if o.get("rank", "").upper() == "A"]
    rang_a_str = "\n".join(f"- {o['label']} [Rang A]" for o in rang_a) or objectives_str

    return (
        f"### ITEM ET RÉFÉRENCE\n\n"
        f"{item_line}\n\n"
        f"### OBJECTIFS DE CONNAISSANCE\n\n"
        f"La question doit évaluer l'un des objectifs officiels de cet item.\n"
        f"Priorité absolue aux objectifs de Rang A.\n\n"
        f"{objectives_str}\n\n"
        f"### OBJECTIF CIBLÉ\n\n"
        f"Sélectionner l'objectif de Rang A le plus pertinent au regard du contenu fourni.\n"
        f"La question doit évaluer exclusivement cet objectif — toute information hors périmètre est à écarter.\n\n"
        f"Objectifs Rang A disponibles :\n"
        f"{rang_a_str}\n\n"
        f"### CONSIGNES DE RÉDACTION\n\n"
        f"Respecter exclusivement les informations et objectifs présents dans le contenu fourni.\n"
        f"Rédiger en français clair, précis et sans ambiguïté.\n"
        f"La question est indépendante (Question Isolée) — aucun contexte clinique narratif n'est requis.\n\n"
        f"{_QCM_GUIDELINES}\n\n"
        f"{_JSON_OUTPUT_BLOCK}\n\n"
        f"### CONTENU SOURCE\n\n"
        f"{{content}}\n\n"
        f"INSTRUCTION FINALE :\n"
        f"Répondez UNIQUEMENT avec un unique objet JSON valide, sans aucun texte en dehors."
    )


def _build_fallback_prompt() -> str:
    return (
        f"À partir du contenu éducatif suivant, générez exactement une question à choix unique "
        f"avec quatre options de réponse (a, b, c, d), dont une seule est correcte.\n\n"
        f"CONSIGNES DE RÉDACTION :\n"
        f"La question doit évaluer la compréhension des idées principales du contenu fourni.\n\n"
        f"{_QCM_GUIDELINES}\n\n"
        f"{_JSON_OUTPUT_BLOCK}\n\n"
        f"CONTENU ÉDUCATIF :\n"
        f"{{content}}\n\n"
        f"INSTRUCTION FINALE :\n"
        f"Répondez UNIQUEMENT avec un unique objet JSON valide, sans aucun texte en dehors."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_prompt(content: str) -> dict:
    """
    Build an enriched Ollama prompt from educational content.

    Returns:
        {
            "prompt": str,           # prompt string with {content} placeholder
            "item_ref": str | None,  # e.g. "Item 222", None if no LISA match
            "objectives_count": int,
        }
    """
    # If MCP server is not configured, return fallback immediately
    mcp_url, _, _ = _cfg()
    if not mcp_url:
        return {
            "prompt": _build_fallback_prompt(),
            "item_ref": None,
            "objectives_count": 0,
        }

    item_ref: Optional[str] = None
    item_label: Optional[str] = None
    objectives: list[dict] = []

    # Step 1A — regex extraction
    search_term = _extract_via_regex(content)
    if search_term:
        item_ref, item_label, objectives = _fetch_objectives(search_term)

    # Step 1B — Ollama fallback if regex gave no SPARQL results
    if not objectives:
        search_term = _extract_via_ollama(content)
        if search_term:
            item_ref, item_label, objectives = _fetch_objectives(search_term)

    # Step 2 — build prompt
    print("###################",objectives)
    if objectives:
        prompt = _build_enriched_prompt(item_ref, item_label, objectives)
    else:
        prompt = _build_fallback_prompt()
        item_ref = None

    return {
        "prompt": prompt,
        "item_ref": item_ref,
        "objectives_count": len(objectives),
    }
