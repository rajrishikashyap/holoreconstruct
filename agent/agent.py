"""
agent.py  --  the Agent's "brain" (raw, from-scratch loop)
==========================================================
An optical-engineer agent built as a plain Python loop -- NO framework. This is
Stage 1 of Phase 4 (the LangGraph version comes later). Building it by hand
first makes the agent loop fully visible:

    perceive (user goal + tool results)
      -> reason (the LLM decides)
        -> act (call a real tool from tools.py)
          -> observe (feed the result back)
            -> repeat until the LLM gives a final answer

It talks to a LOCAL model via Ollama (no API key, no cost). The model decides
WHICH tool to call and with WHAT arguments; our loop actually runs the tool and
returns the result. That division is the whole point of tool-calling agents.

Prereqs:
    * Ollama installed + running (ollama.com)
    * a tool-capable model pulled, e.g.:  ollama pull llama3.2:3b
    * pip install ollama
    * models/generator.pt present (tools reconstruct with it)

Run:
    python agent.py
    python agent.py --model llama3.2:1b        # smaller model
    python agent.py --ask "the reconstruction looks blurry, fix the focus"
"""

import json
import argparse

import ollama

from tools import TOOLS


SYSTEM_PROMPT = """You are an optical-engineering assistant embedded in a \
holographic reconstruction tool. Reconstructions can look blurry when the \
propagation distance z is set wrong. You have tools to sweep z and measure \
sharpness. When the user reports blur or a focus problem, use run_z_sweep to \
find the sharpest z, then report the best z clearly. Be concise and precise. \
Only call a tool when it helps; when you have the answer, state it plainly."""

# Build the tool specs in the shape Ollama expects.
OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": spec["description"],
            "parameters": spec["parameters"],
        },
    }
    for name, spec in TOOLS.items()
]


def run_agent(user_message, model="llama3.2:3b", max_steps=6, verbose=True):
    """
    The core agent loop. Returns the final text answer.

    Each iteration:
      1. send the running conversation to the model (with tool specs)
      2. if the model asks for a tool -> run it, append the result, loop
      3. if the model returns plain text -> that's the final answer, stop
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for step in range(max_steps):
        # --- REASON: ask the model what to do next ---
        resp = ollama.chat(
            model=model,
            messages=messages,
            tools=OLLAMA_TOOLS,
            options={"num_ctx": 8192, "temperature": 0.2},  # raise ctx window!
        )
        msg = resp["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            # --- FINISH: no tool requested -> this is the final answer ---
            if verbose:
                print(f"\n[final answer after {step} tool step(s)]")
            return msg.get("content", "").strip()

        # --- ACT + OBSERVE: run each requested tool, feed results back ---
        for call in tool_calls:
            fname = call["function"]["name"]
            fargs = call["function"].get("arguments", {}) or {}
            if isinstance(fargs, str):
                try:
                    fargs = json.loads(fargs)
                except Exception:
                    fargs = {}

            if verbose:
                print(f"[step {step}] model calls: {fname}({fargs})")

            if fname not in TOOLS:
                result = {"error": f"unknown tool {fname}"}
            else:
                try:
                    result = TOOLS[fname]["fn"](**fargs)
                except Exception as e:
                    result = {"error": str(e)}

            if verbose:
                print(f"          tool result: {result}")

            messages.append({
                "role": "tool",
                "content": json.dumps(result),
            })

    return "(stopped: reached max steps without a final answer)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--ask", default="The reconstruction looks blurry near the "
                                      "edges. Can you find a better focus distance? "
                                      "Try z between 0.5 and 1.5 mm.")
    args = ap.parse_args()

    print(f"model: {args.model}")
    print(f"user:  {args.ask}\n")
    answer = run_agent(args.ask, model=args.model)
    print("\n" + "=" * 60)
    print("AGENT:", answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
