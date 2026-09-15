"""
main.py  --  FastAPI backend for HoloReconstruct (deployment build)
===================================================================
Serves the trained generator(s) behind a simple HTTP API. This build supports
TWO models (MNIST digits + BloodMNIST cells) and lets the frontend pick which.

Endpoints:
  GET  /            -> serves the frontend
  GET  /health      -> status + which models are loaded
  GET  /models      -> list of available models for the UI toggle
  POST /reconstruct -> upload an image (+ optional ?model=), get back
                       {input, hologram, reconstruction} as base64 PNGs
  POST /agent       -> optical-engineer agent (live if Ollama present,
                       otherwise returns a friendly "demo mode" message)

Model files expected in models/:
  generator_mnist.pt   and/or   generator_blood.pt
  (falls back to generator.pt if the named ones are absent)

On Hugging Face Spaces there is no GPU and no Ollama; the app auto-detects CPU
and the agent replies in demo mode. Reconstruction is fully live.
"""

import io
import os
import sys
import base64

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "core")
BACKEND = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CORE)
sys.path.insert(0, BACKEND)

import torch
from propagation_asm import image_to_hologram
from networks import Generator
from dataset import HoloConfig, _complex_to_channels, _normalise_per_sample

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HC = HoloConfig()
HOLO_C = 1 if HC.representation == "intensity" else 2
LATENT_DIM = 128
MODELS_DIR = os.path.join(ROOT, "models")

# registry of models the UI can offer: key -> (filename, label, hint)
MODEL_SPECS = {
    "mnist": ("generator_mnist.pt", "MNIST digits",
              "handwritten digits reconstruct crisply"),
    "blood": ("generator_blood.pt", "Blood cells",
              "real blood-cell microscopy (the medical result)"),
}
loaded = {}  # key -> Generator


def _maybe_download(fname):
    """
    If models/<fname> is missing but an env var points to it, download it.
    Env var names: MODEL_URL_MNIST, MODEL_URL_BLOOD (set these in Render).
    Lets you host weights on Hugging Face / a release / a direct link.
    """
    dest = os.path.join(MODELS_DIR, fname)
    if os.path.exists(dest):
        return dest
    env_key = "MODEL_URL_" + fname.replace("generator_", "").replace(".pt", "").upper()
    url = os.environ.get(env_key)
    if not url:
        return dest  # nothing to download; caller will skip if absent
    try:
        import urllib.request
        os.makedirs(MODELS_DIR, exist_ok=True)
        print(f"[dl] fetching {fname} from {env_key}...")
        urllib.request.urlretrieve(url, dest)
        print(f"[dl] saved {fname} ({os.path.getsize(dest)//1024} KB)")
    except Exception as e:
        print(f"[dl] failed for {fname}: {e}")
    return dest


def _load_one(path):
    g = Generator(HOLO_C, latent_dim=LATENT_DIM, image_size=HC.image_size).to(DEVICE)
    g.load_state_dict(torch.load(path, map_location=DEVICE))
    g.eval()
    return g


def load_models():
    """Load whichever model files are present (downloading first if a URL is set)."""
    for key, (fname, _label, _hint) in MODEL_SPECS.items():
        p = _maybe_download(fname)
        if os.path.exists(p):
            loaded[key] = _load_one(p)
            print(f"[ok] '{key}' model loaded from {fname} on {DEVICE}")
    # legacy single-file fallback
    if not loaded:
        p = os.path.join(MODELS_DIR, "generator.pt")
        if os.path.exists(p):
            loaded["default"] = _load_one(p)
            print(f"[ok] default model loaded from generator.pt on {DEVICE}")
        else:
            print("[warn] no model files found; /reconstruct will 503")


def _pick(model_key):
    """Resolve a requested model key to a loaded generator."""
    if model_key and model_key in loaded:
        return loaded[model_key]
    # default preference: blood, then mnist, then whatever loaded
    for k in ("blood", "mnist", "default"):
        if k in loaded:
            return loaded[k]
    return None


# --- image helpers ---
def pil_to_gray_tensor(pil_img, size):
    pil_img = pil_img.convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def tensor_to_png_datauri(t):
    arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def hologram_preview_tensor(field):
    inten = field.abs() ** 2
    return (inten - inten.min()) / (inten.max() - inten.min()).clamp_min(1e-8)


app = FastAPI(title="HoloReconstruct API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup():
    load_models()
    try:
        from agent_service import share_model
        g = _pick(None)
        if g is not None:
            share_model(g, HC, DEVICE)
    except Exception as e:
        print(f"[info] agent model-share skipped: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(loaded.keys()), "device": DEVICE}


@app.get("/models")
def models():
    """What the UI toggle should show: only models actually loaded."""
    out = []
    for key, (_f, label, hint) in MODEL_SPECS.items():
        if key in loaded:
            out.append({"key": key, "label": label, "hint": hint})
    if not out and "default" in loaded:
        out.append({"key": "default", "label": "Model", "hint": ""})
    return {"models": out, "default": (out[0]["key"] if out else None)}


@app.post("/reconstruct")
async def reconstruct(file: UploadFile = File(...),
                      model: str = Query(default=None)):
    gen = _pick(model)
    if gen is None:
        raise HTTPException(status_code=503, detail="no model loaded")

    try:
        raw = await file.read()
        pil = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="could not read image file")

    clean = pil_to_gray_tensor(pil, HC.image_size).to(DEVICE)
    with torch.no_grad():
        field = image_to_hologram(clean, HC.z, HC.wavelength, HC.pixel_size)
        holo = _complex_to_channels(field, HC.representation)
        holo = _normalise_per_sample(holo).unsqueeze(0).float().to(DEVICE)
        z0 = torch.zeros(1, LATENT_DIM, device=DEVICE)
        recon = gen(z0, holo)[0, 0]

    return {
        "input": tensor_to_png_datauri(clean),
        "hologram": tensor_to_png_datauri(hologram_preview_tensor(field)),
        "reconstruction": tensor_to_png_datauri(recon),
        "meta": {"device": DEVICE, "size": HC.image_size, "model": model or "auto"},
    }


class AgentRequest(BaseModel):
    message: str
    model: str = "llama3.2:3b"


@app.post("/agent")
def agent(req: AgentRequest):
    """Live agent if Ollama+LangGraph are available; else a demo-mode reply."""
    try:
        from agent_service import run_agent_web, _ensure_agent
        if _ensure_agent():
            return run_agent_web(req.message, model=req.model)
    except Exception:
        pass
    return {
        "ok": True,
        "answer": ("Agent runs locally with Ollama, so it is not live on this "
                   "hosted demo. Clone the repo and run it locally to chat with "
                   "the optical-engineer agent, which sweeps the propagation "
                   "distance z to auto-focus reconstructions. See the README for "
                   "a recorded demo."),
        "trajectory": [],
        "demo_mode": True,
    }


FRONTEND_DIR = os.path.join(ROOT, "frontend")


@app.get("/")
def index():
    idx = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(idx) if os.path.exists(idx) else {"msg": "API running, see /docs"}


if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
