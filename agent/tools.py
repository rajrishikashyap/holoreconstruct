"""
tools.py  --  the Agent's "hands"
=================================
Real Python functions the agent can call. The LLM never runs these itself; it
emits a structured request naming a tool + args, and agent.py executes the
matching function here and feeds the result back.

Core idea (the optical-engineer agent): when a reconstruction looks blurry, the
usual cause is a wrong propagation distance z. These tools let the agent sweep
z, measure sharpness, and pick the best value -- automatically.

Each tool:
  * takes plain JSON-friendly args (numbers, strings)
  * returns a plain JSON-friendly dict
so the loop can hand results straight back to the model.

Requires the core/ modules (propagation_asm, networks, dataset) importable.
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "core")
sys.path.insert(0, CORE)

import torch
from propagation_asm import image_to_hologram, propagate
from networks import Generator
from dataset import HoloConfig, _complex_to_channels, _normalise_per_sample

# --- shared state loaded once ---
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_HC = HoloConfig()
_HOLO_C = 1 if _HC.representation == "intensity" else 2
_GEN = None
# a fixed test image the agent experiments on (a simple centered block)
_TEST_IMG = None


def _lazy_init():
    """Load the generator + a test image once, on first tool use."""
    global _GEN, _TEST_IMG
    if _GEN is None:
        path = os.path.join(ROOT, "models", "generator.pt")
        g = Generator(_HOLO_C, latent_dim=128, image_size=_HC.image_size).to(_DEVICE)
        g.load_state_dict(torch.load(path, map_location=_DEVICE))
        g.eval()
        _GEN = g
    if _TEST_IMG is None:
        img = torch.zeros((_HC.image_size, _HC.image_size), device=_DEVICE)
        s = _HC.image_size
        img[s//3:2*s//3, 5*s//12:7*s//12] = 1.0   # a vertical bar
        _TEST_IMG = img


def _sharpness(img_2d):
    """
    Variance of the Laplacian -- a standard focus/sharpness metric.
    Higher = sharper. A blurry reconstruction scores low.
    """
    a = img_2d.detach().cpu().numpy().astype(np.float64)
    lap = (
        -4 * a
        + np.roll(a, 1, 0) + np.roll(a, -1, 0)
        + np.roll(a, 1, 1) + np.roll(a, -1, 1)
    )
    return float(lap.var())


def _reconstruct_at_z(z):
    """Run the pipeline (ASM at distance z -> network) and return recon + sharpness."""
    _lazy_init()
    with torch.no_grad():
        field = image_to_hologram(_TEST_IMG, z, _HC.wavelength, _HC.pixel_size)
        holo = _complex_to_channels(field, _HC.representation)
        holo = _normalise_per_sample(holo).unsqueeze(0).float().to(_DEVICE)
        z0 = torch.zeros(1, 128, device=_DEVICE)
        recon = _GEN(z0, holo)[0, 0]
    return recon, _sharpness(recon)


# --------------------------------------------------------------------------- #
#  TOOLS (these are what the agent can call)
# --------------------------------------------------------------------------- #
def measure_sharpness(z_mm: float) -> dict:
    """Reconstruct at a given propagation distance (in mm) and report sharpness."""
    z = float(z_mm) * 1e-3
    _, sharp = _reconstruct_at_z(z)
    return {"z_mm": round(float(z_mm), 4), "sharpness": round(sharp, 6)}


def run_z_sweep(start_mm: float, end_mm: float, steps: int = 5) -> dict:
    """
    Sweep propagation distance z from start_mm to end_mm over N steps,
    measuring reconstruction sharpness at each. Returns every reading plus the
    best (sharpest) z. This is the agent's main diagnostic for blur.
    """
    steps = max(2, min(int(steps), 15))   # guard rails
    zs = np.linspace(float(start_mm), float(end_mm), steps)
    readings = []
    for z_mm in zs:
        _, sharp = _reconstruct_at_z(float(z_mm) * 1e-3)
        readings.append({"z_mm": round(float(z_mm), 4), "sharpness": round(sharp, 6)})
    best = max(readings, key=lambda r: r["sharpness"])
    return {"readings": readings, "best_z_mm": best["z_mm"],
            "best_sharpness": best["sharpness"]}


def get_current_config() -> dict:
    """Report the current optical parameters the pipeline is using."""
    return {
        "wavelength_nm": round(_HC.wavelength * 1e9, 1),
        "pixel_size_um": round(_HC.pixel_size * 1e6, 2),
        "z_mm": round(_HC.z * 1e3, 3),
        "representation": _HC.representation,
        "device": _DEVICE,
    }


# registry the agent loop reads: name -> (function, description, schema)
TOOLS = {
    "run_z_sweep": {
        "fn": run_z_sweep,
        "description": "Sweep propagation distance z (mm) over a range and find "
                       "the sharpest reconstruction. Use when the image looks blurry.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_mm": {"type": "number", "description": "start distance in mm"},
                "end_mm":   {"type": "number", "description": "end distance in mm"},
                "steps":    {"type": "integer", "description": "how many points (2-15)"},
            },
            "required": ["start_mm", "end_mm"],
        },
    },
    "measure_sharpness": {
        "fn": measure_sharpness,
        "description": "Measure reconstruction sharpness at one specific z (mm).",
        "parameters": {
            "type": "object",
            "properties": {"z_mm": {"type": "number", "description": "distance in mm"}},
            "required": ["z_mm"],
        },
    },
    "get_current_config": {
        "fn": get_current_config,
        "description": "Get the current optical parameters (wavelength, pixel size, z).",
        "parameters": {"type": "object", "properties": {}},
    },
}


# --------------------------------------------------------------------------- #
#  self-test (no LLM needed) -- run `python tools.py`
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("device:", _DEVICE)
    print("config:", get_current_config())
    print("single measure @1.0mm:", measure_sharpness(1.0))
    print("sweep 0.5 -> 1.5mm:")
    res = run_z_sweep(0.5, 1.5, steps=5)
    for r in res["readings"]:
        print(f"   z={r['z_mm']}mm  sharpness={r['sharpness']}")
    print(f"   -> best z = {res['best_z_mm']}mm (sharpness {res['best_sharpness']})")
