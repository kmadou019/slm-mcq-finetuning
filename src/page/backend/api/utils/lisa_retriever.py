"""
Thin wrapper around agent_retrieve.agent.retrieve_lisa_objectives.
Adds sys.path resolution and failure isolation so generation.py is not impacted
if GraphDB / Ollama are unreachable.
"""
import sys
from pathlib import Path

# project root is 5 levels up from this file
_PROJECT_ROOT = Path(__file__).parents[5]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def retrieve_lisa_objectives_safe(content: str) -> str:
    """Return LISA context string, or '' on any failure (network, timeout, etc.)."""
    try:
        from agent_retrieve.agent import retrieve_lisa_objectives
        return retrieve_lisa_objectives(content)
    except Exception as e:
        print(f"[lisa_retriever] agent failed, skipping LISA enrichment: {e}")
        return ""
