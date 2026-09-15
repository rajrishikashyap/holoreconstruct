"""
agent_service.py  --  the agent, wired for the web dashboard
============================================================
Wraps the LangGraph optical-engineer agent so the FastAPI backend can call it
and return a structured result the UI can render:

    {
      "answer": "...final text...",
      "trajectory": [ {"tool": "run_z_sweep", "args": {...}, "result": {...}}, ... ],
      "best_z_mm": 1.21          # convenience, pulled from a sweep if one ran
    }

Design ("best of both"):
  * engine = LangGraph (robust state machine)
  * tools  = the same core functions (from agent/tools.py), shared
  * the backend's already-loaded generator is REUSED (no second model load)

This module is imported by backend/main.py. It degrades gracefully: if Ollama
or LangGraph isn't available, the /agent endpoint reports that clearly instead
of crashing the whole server.
"""

import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "agent"))

# Whether the agent stack is importable is checked lazily so the web app still
# serves reconstructions even if the agent deps aren't installed.
_AGENT_READY = None
_AGENT_ERR = None
_build_agent = None


def _ensure_agent():
    """Import the LangGraph agent lazily; cache success/failure."""
    global _AGENT_READY, _AGENT_ERR, _build_agent
    if _AGENT_READY is not None:
        return _AGENT_READY
    try:
        # importing agent_langgraph pulls in langgraph + langchain_ollama + tools
        from agent_langgraph import build_agent
        _build_agent = build_agent
        _AGENT_READY = True
    except Exception as e:
        _AGENT_READY = False
        _AGENT_ERR = str(e)
    return _AGENT_READY


def share_model(generator, holo_config, device):
    """
    Let the agent's tools reuse the backend's already-loaded generator instead
    of loading a second copy. Called once by main.py at startup.
    """
    try:
        import tools as agent_tools
        agent_tools._GEN = generator
        agent_tools._HC = holo_config
        agent_tools._DEVICE = device
        # the test image is built lazily on first use; leave it
    except Exception:
        pass  # if tools import fails here, _ensure_agent will report it later


def run_agent_web(user_message, model="llama3.2:3b"):
    """
    Run the LangGraph agent and return a UI-friendly structured result.
    Never raises to the caller -- errors come back in the dict.
    """
    if not _ensure_agent():
        return {
            "ok": False,
            "answer": None,
            "error": f"Agent unavailable: {_AGENT_ERR}. "
                     f"Install with: pip install langgraph langchain-ollama, "
                     f"and ensure Ollama is running with the model pulled.",
            "trajectory": [],
        }

    try:
        from langchain_core.messages import HumanMessage
        agent = _build_agent(model=model)
        result = agent.invoke({"messages": [HumanMessage(content=user_message)]})

        # walk the message trajectory: pair each tool call with its result
        trajectory = []
        pending = {}
        best_z = None
        for m in result["messages"]:
            calls = getattr(m, "tool_calls", None)
            if calls:
                for tc in calls:
                    pending[tc.get("id") or tc["name"]] = {
                        "tool": tc["name"], "args": tc.get("args", {})
                    }
            elif getattr(m, "type", None) == "tool":
                # tool result message
                try:
                    parsed = json.loads(m.content)
                except Exception:
                    parsed = m.content
                # attach to the most recent pending call
                key = getattr(m, "tool_call_id", None)
                entry = pending.pop(key, None) if key else None
                if entry is None and pending:
                    entry = pending.pop(next(iter(pending)))
                if entry is None:
                    entry = {"tool": "?", "args": {}}
                entry["result"] = parsed
                trajectory.append(entry)
                if isinstance(parsed, dict) and "best_z_mm" in parsed:
                    best_z = parsed["best_z_mm"]

        final = result["messages"][-1]
        answer = getattr(final, "content", "").strip()

        return {
            "ok": True,
            "answer": answer,
            "trajectory": trajectory,
            "best_z_mm": best_z,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "answer": None,
            "error": f"Agent run failed: {e}",
            "trajectory": [],
        }
