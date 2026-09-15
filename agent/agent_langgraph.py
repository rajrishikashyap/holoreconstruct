"""
agent_langgraph.py  --  the Agent as a LangGraph state machine (Stage 2)
========================================================================
Same optical-engineer agent as agent.py, but the hand-written perceive→reason→
act→observe loop is replaced by an explicit LangGraph STATE MACHINE.

Compare this file with agent.py to see exactly what the framework abstracts:

    agent.py (raw)                 agent_langgraph.py (framework)
    --------------                 ------------------------------
    a for-loop                     a compiled StateGraph
    if msg.tool_calls: run them    a ToolNode + conditional edge
    manual message list            managed state (MessagesState)
    you drive the loop             graph.invoke() drives it

The graph:

        (START)
           │
           ▼
      ┌─────────┐   no tool needed
      │ reason  │ ───────────────────► (END)
      │ (LLM)   │
      └────┬────┘
           │ tool needed
           ▼
      ┌─────────┐
      │  tools  │ ──┐
      └─────────┘   │ loop back to reason with the result
           ▲────────┘

Same tools as the raw agent (imported from tools.py), same local Ollama model,
so any behaviour difference is the framework's doing, not the logic's.

Prereqs (in addition to the raw agent's):
    pip install langgraph langchain-ollama

Run:
    python agent_langgraph.py
    python agent_langgraph.py --model llama3.2:1b
    python agent_langgraph.py --ask "find the sharpest z between 0.5 and 2.0 mm"
"""

import argparse

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# reuse the SAME underlying functions the raw agent used
from tools import run_z_sweep as _sweep
from tools import measure_sharpness as _measure
from tools import get_current_config as _config


# --------------------------------------------------------------------------- #
#  1. Wrap the existing functions as LangChain tools (@tool adds the schema)
# --------------------------------------------------------------------------- #
@tool
def run_z_sweep(start_mm: float, end_mm: float, steps: int = 5) -> dict:
    """Sweep propagation distance z (mm) over a range and find the sharpest
    reconstruction. Use this when the image looks blurry or out of focus."""
    return _sweep(start_mm, end_mm, steps)


@tool
def measure_sharpness(z_mm: float) -> dict:
    """Measure reconstruction sharpness at one specific propagation distance z (mm)."""
    return _measure(z_mm)


@tool
def get_current_config() -> dict:
    """Get the current optical parameters (wavelength, pixel size, z, device)."""
    return _config()


TOOLS = [run_z_sweep, measure_sharpness, get_current_config]

SYSTEM_PROMPT = (
    "You are an optical-engineering assistant embedded in a holographic "
    "reconstruction tool. Reconstructions can look blurry when the propagation "
    "distance z is wrong. Use run_z_sweep to find the sharpest z, then report "
    "the best z clearly and concisely. Only call a tool when it helps."
)


# --------------------------------------------------------------------------- #
#  2. Build the state machine
# --------------------------------------------------------------------------- #
def build_agent(model="llama3.2:3b"):
    # the LLM, told about the tools (bind_tools attaches the schemas)
    llm = ChatOllama(model=model, temperature=0.2, num_ctx=8192)
    llm_with_tools = llm.bind_tools(TOOLS)

    # NODE: "reason" -- the LLM looks at the conversation and either answers
    # or emits tool calls. This is the exact analogue of the raw loop's
    # "ollama.chat(...)" step.
    def reason(state: MessagesState):
        msgs = state["messages"]
        # inject the system prompt once, at the front
        if not msgs or msgs[0].type != "system":
            from langchain_core.messages import SystemMessage
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
        return {"messages": [llm_with_tools.invoke(msgs)]}

    # NODE: "tools" -- runs whatever tool the LLM asked for. ToolNode is the
    # framework's version of the raw loop's "for call in tool_calls: run it".
    tool_node = ToolNode(TOOLS)

    # wire the graph
    g = StateGraph(MessagesState)
    g.add_node("reason", reason)
    g.add_node("tools", tool_node)
    g.add_edge(START, "reason")
    # conditional edge: if the LLM asked for a tool -> "tools", else -> END.
    # tools_condition is a prebuilt that inspects the last message for tool calls.
    g.add_conditional_edges("reason", tools_condition, {"tools": "tools", END: END})
    # after running tools, always loop back to reason with the results
    g.add_edge("tools", "reason")

    return g.compile()


def run(user_message, model="llama3.2:3b", verbose=True):
    agent = build_agent(model)
    from langchain_core.messages import HumanMessage
    result = agent.invoke({"messages": [HumanMessage(content=user_message)]})

    if verbose:
        # show the trajectory: which tools were called, with what
        for m in result["messages"]:
            if getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    print(f"[tool call] {tc['name']}({tc['args']})")
            elif m.type == "tool":
                print(f"[tool result] {m.content}")
    # the final answer is the last AI message with no tool calls
    final = result["messages"][-1]
    return final.content.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--ask", default="The reconstruction looks blurry near the "
                                     "edges. Can you find a better focus distance? "
                                     "Try z between 0.5 and 1.5 mm.")
    args = ap.parse_args()

    print(f"model: {args.model}  (LangGraph version)")
    print(f"user:  {args.ask}\n")
    answer = run(args.ask, model=args.model)
    print("\n" + "=" * 60)
    print("AGENT:", answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
