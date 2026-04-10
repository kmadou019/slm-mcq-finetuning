"""
LangGraph SPARQL Query Agent — ReAct style
LLM: qwen3.5:3b via Ollama
Triplestore: GraphDB at http://localhost:7200/repositories/lisa

Each iteration the LLM either:
  {"action": "query", "sparql": "..."}   → run one focused query, loop back
  {"action": "answer"}                   → enough info collected, synthesize answer
"""

import json
import os
import re
import requests
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()

SPARQL_ENDPOINT  = os.getenv("GRAPHDB_ENDPOINT",  "http://localhost:7200/repositories/lisa")
OLLAMA_CHAT_URL  = os.getenv("OLLAMA_CHAT_URL",   "http://localhost:11434/api/chat")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL",      "qwen3.5:35b")
MAX_ATTEMPTS     = 8

SYSTEM_PROMPT = """\
Tu es un assistant expert GraphDB et SPARQL qui répond aux questions en interrogeant \
une base de connaissances médicales (ontologie LISA).

RÈGLE ABSOLUE : tu DOIS toujours utiliser des requêtes SPARQL pour répondre. \
N'utilise JAMAIS tes connaissances internes sans interroger la base.

OUTPUT FORMAT — réponds UNIQUEMENT avec un objet JSON, sans markdown :
  {"action": "query", "sparql": "..."}   → exécute cette requête SPARQL
  {"action": "answer"}                   → tu as assez d'informations pour répondre

STRATÉGIE D'EXPLORATION OBLIGATOIRE (5 à 8 requêtes minimum) :

1. PREMIÈRE REQUÊTE — découverte du schéma :
   SELECT ?class (COUNT(?s) AS ?n) WHERE { ?s a ?class } GROUP BY ?class ORDER BY DESC(?n) LIMIT 20

2. DEUXIÈME REQUÊTE — recherche du terme principal (labels + commentaires) :
   Utilise FILTER regex insensible à la casse sur rdfs:label et rdfs:comment.

3. TROISIÈME REQUÊTE — variantes orthographiques et synonymes :
   Essaie des préfixes/suffixes plus larges (ex: "retin" pour rétinopathie, rétine, retinal…).

4. QUATRIÈME REQUÊTE — exploration d'une URI trouvée :
   SELECT ?p ?o WHERE { <URI_trouvée> ?p ?o } LIMIT 50

5. CINQUIÈME REQUÊTE — relations vers d'autres entités :
   SELECT ?relation ?related ?relatedLabel WHERE {
     <URI> ?relation ?related .
     OPTIONAL { ?related rdfs:label ?relatedLabel }
   }

6. REQUÊTES SUIVANTES — approfondissement selon les résultats obtenus.

RÈGLES SPARQL CRITIQUES :
❗ TOUJOURS utiliser une variable comme sujet : `?item a lisa:KnowledgeItem ; rdfs:label ?label .`
   JAMAIS une URI de classe comme sujet : `lisa:KnowledgeItem rdfs:label ?label` est INCORRECT.
❗ LIMIT 20 sur toutes les requêtes sauf exploration de propriétés d'une URI (LIMIT 50).
❗ Pas de UNION ni d'OPTIONAL imbriqués profonds — décompose en plusieurs requêtes successives.
❗ Ne t'arrête JAMAIS après 1-2 requêtes vides — essaie des variantes.
❗ Utilise les résultats d'une requête pour guider la suivante.
❗ Utilise TOUJOURS l'orthographe française accentuée dans les recherches : 'rétinopathie' et non 'retinopathie', 'détresse' et non 'detresse', etc.
❗ Syntaxe FILTER : le FILTER doit être DANS les accolades du WHERE, jamais après la dernière accolade fermante.

PRÉFIXES :
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX lisa: <https://uness.fr/lisa/ontology/>

CLASSES PRINCIPALES :
- lisa:KnowledgeObjective  (objectifs de connaissance)
- lisa:LearningOutcome     (résultats d'apprentissage)
- lisa:KnowledgeItem       (items de connaissance)
- lisa:StartingSituation   (situations de départ)
- lisa:College             (collèges)
- lisa:Person

PROPRIÉTÉS PRINCIPALES :
- rdfs:label               (nom/libellé)
- rdfs:comment             (description détaillée)
- lisa:rank                (niveau d'importance)
- lisa:family              (famille thématique)
- lisa:hasRelatedKnowledgeItems
- lisa:hasKnowledgeObjective
- lisa:hasLearningOutcome
"""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    history: list          # [{"query": str, "results": str | "ERROR: ..."}]
    current_query: str
    current_results: str
    error_message: str
    final_answer: str
    attempts: int
    next_action: str       # "query" | "answer" | "error"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm(prompt: str, think: bool = False) -> str:
    """Blocking call for the reasoner — needs full response before routing."""
    payload = {
        "model": OLLAMA_MODEL,
        "think": think,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    }
    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=300)
    resp.raise_for_status()
    msg = resp.json()["message"]
    if think and msg.get("thinking"):
        _log("REASONING", msg["thinking"])
    return msg["content"].strip()


def _llm_stream(prompt: str) -> str:
    """Streaming call for the answer node — prints tokens live."""
    payload = {
        "model": OLLAMA_MODEL,
        "think": False,
        "stream": True,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }
    resp = requests.post(OLLAMA_CHAT_URL, json=payload, stream=True, timeout=300)
    resp.raise_for_status()
    parts = []
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        token = chunk.get("message", {}).get("content", "")
        print(token, end="", flush=True)
        parts.append(token)
        if chunk.get("done"):
            break
    print()
    return "".join(parts).strip()


def _parse_action(text: str) -> dict:
    """Extract the JSON action from LLM output."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    # Find first {...}
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"action": "error", "raw": text}


def _log(label: str, content: str = "") -> None:
    sep = "─" * 60
    print(f"\n{sep}\n[{label}]\n{sep}")
    if content:
        print(content)


def _build_prompt(state: AgentState) -> str:
    parts = [f"Question: {state['question']}\n"]
    for i, step in enumerate(state["history"], 1):
        parts.append(f"--- Query {i} ---\n{step['query']}\nResults:\n{step['results']}")
    parts.append("\nWhat do you do next? Output JSON only.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def reasoner(state: AgentState) -> dict:
    """LLM decides: run a new query or stop."""
    attempt = state["attempts"] + 1
    _log(f"REASONER  step {attempt}/{MAX_ATTEMPTS}")

    prompt = _build_prompt(state)
    raw = _llm(prompt, think=True)
    action = _parse_action(raw)

    _log("LLM DECISION", json.dumps(action, ensure_ascii=False, indent=2))

    if action.get("action") == "query":
        return {
            "current_query": action.get("sparql", "").strip(),
            "attempts": attempt,
            "error_message": "",
            "next_action": "query",
        }
    elif action.get("action") == "answer":
        return {
            "current_query": "",
            "attempts": attempt,
            "error_message": "",
            "next_action": "answer",
        }
    else:
        return {
            "current_query": "",
            "attempts": attempt,
            "error_message": f"Could not parse LLM output: {raw[:200]}",
            "next_action": "error",
        }


def executor(state: AgentState) -> dict:
    _log("EXECUTOR", state["current_query"])
    try:
        resp = requests.post(
            SPARQL_ENDPOINT,
            data={"query": state["current_query"]},
            headers={"Accept": "application/sparql-results+json"},
            timeout=30,
        )
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])
        if not bindings:
            result_str = "(no results)"
        else:
            result_str = json.dumps(bindings, ensure_ascii=False, indent=2)
        _log("RESULT", result_str)
        new_step = {"query": state["current_query"], "results": result_str}
        return {
            "history": state["history"] + [new_step],
            "current_results": result_str,
            "error_message": "",
        }
    except requests.HTTPError as e:
        msg = f"ERROR: HTTP {e.response.status_code}: {e.response.text[:300]}"
        _log("ERROR", msg)
        new_step = {"query": state["current_query"], "results": msg}
        return {
            "history": state["history"] + [new_step],
            "current_results": "",
            "error_message": msg,
        }
    except Exception as e:
        msg = f"ERROR: {e}"
        _log("ERROR", msg)
        new_step = {"query": state["current_query"], "results": msg}
        return {
            "history": state["history"] + [new_step],
            "current_results": "",
            "error_message": msg,
        }


def answer_generator(state: AgentState) -> dict:
    _log("ANSWER GENERATOR")
    summary_parts = [f"Question: {state['question']}\n\nDonnées collectées :"]
    for i, step in enumerate(state["history"], 1):
        summary_parts.append(f"\nRequête {i}:\n{step['query']}\nRésultats:\n{step['results']}")
    summary_parts.append(
        "\nProduis une réponse structurée dans la même langue que la question :\n"
        "## Résultats de l'exploration\n[résume les découvertes]\n"
        "## Réponse complète\n[réponse détaillée basée uniquement sur les données trouvées]\n"
        "## Requêtes SPARQL utilisées\n[liste numérotée]"
    )
    prompt = "\n".join(summary_parts)
    answer = _llm_stream(prompt)
    return {"final_answer": answer}


def fail_node(state: AgentState) -> dict:
    msg = f"Failed after {state['attempts']} steps. Last error: {state.get('error_message', 'unknown')}"
    _log("FAILED", msg)
    return {"final_answer": msg}


# ---------------------------------------------------------------------------
# Routing  (runs after reasoner, before executor or answer)
# ---------------------------------------------------------------------------

def router(state: AgentState) -> str:
    if state["next_action"] == "answer":
        return "answer"
    if state["next_action"] == "query":
        if state["attempts"] < MAX_ATTEMPTS:
            return "executor"
        # Limit reached but LLM still wants to query — answer with what we have
        return "answer"
    # error or unparseable
    return "fail"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("reasoner",  reasoner)
workflow.add_node("executor",  executor)
workflow.add_node("answer",    answer_generator)
workflow.add_node("fail",      fail_node)

workflow.add_edge(START, "reasoner")
workflow.add_conditional_edges(
    "reasoner",
    router,
    {"executor": "executor", "answer": "answer", "fail": "fail", "reasoner": "reasoner"},
)
workflow.add_edge("executor", "reasoner")   # always loop back to reason
workflow.add_edge("answer", END)
workflow.add_edge("fail", END)

app = workflow.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(question: str) -> str:
    result = app.invoke({
        "question": question,
        "history": [],
        "current_query": "",
        "current_results": "",
        "error_message": "",
        "final_answer": "",
        "attempts": 0,
        "next_action": "query",
    })
    return result["final_answer"]


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "List all knowledge items and their labels."
    run(q)  # answer already printed by streaming
