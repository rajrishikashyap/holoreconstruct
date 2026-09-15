"""
main.py  --  FastAPI backend for HoloReconstruct
=================================================
Serves the trained generator (from core/) behind a simple HTTP API.

Endpoints:
  GET  /            -> serves the frontend (frontend/index.html)
  GET  /health      -> {"status", "model_loaded", "device", ...}
  POST /reconstruct -> upload an image; returns {input, hologram, reconstruction}
                       as base64 PNGs. Demo mode: the image is turned into a
                       synthetic hologram via ASM, then reconstructed.

Serving rules that matter:
  * model loaded ONCE at startup (not per request)
  * inference under torch.no_grad() + model.eval()
  * preprocessing reuses core/dataset.py so the input matches training exactly

Run (from the project root, with your conda env active):
    uvicorn backend.main:app --reload --port 8000
Then open http://localhost:8000  (frontend) or /docs (auto API UI).
"""

import io
import os
import sys
import base64

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

# --- make the core/ modules importable ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "core")
BACKEND = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CORE)
sys.path.insert(0, BACKEND)   # so `import agent_service` works when run as backend.main

import torch
from propagation_asm import image_to_hologram
from networks import Generator
from dataset import HoloConfig, _complex_to_channels, _normalise_per_sample


# --------------------------------------------------------------------------- #
#  Model loading (once, at startup)
# --------------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HC = HoloConfig()
HOLO_C = 1 if HC.representation == "intensity" else 2
MODEL_PATH = os.path.join(ROOT, "models", "generator.pt")
LATENT_DIM = 128

generator = None


def load_model():
    global generator
    if not os.path.exists(MODEL_PATH):
        print(f"[warn] {MODEL_PATH} not found -- /reconstruct will 503 until present")
        return
    g = Generator(HOLO_C, latent_dim=LATENT_DIM, image_size=HC.image_size).to(DEVICE)
    g.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    g.eval()
    generator = g
    print(f"[ok] generator loaded from {MODEL_PATH} on {DEVICE}")


# --------------------------------------------------------------------------- #
#  Image helpers
# --------------------------------------------------------------------------- #
def pil_to_gray_tensor(pil_img, size):
    pil_img = pil_img.convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def tensor_to_png_datauri(t):
    arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def hologram_preview_tensor(field):
    inten = (field.abs() ** 2)
    inten = (inten - inten.min()) / (inten.max() - inten.min()).clamp_min(1e-8)
    return inten


# --------------------------------------------------------------------------- #
#  App
# --------------------------------------------------------------------------- #
app = FastAPI(title="HoloReconstruct API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    load_model()
    # let the agent's tools reuse this already-loaded model (no second load)
    try:
        from agent_service import share_model
        if generator is not None:
            share_model(generator, HC, DEVICE)
    except Exception as e:
        print(f"[warn] agent model-share skipped: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": generator is not None,
        "device": DEVICE,
        "representation": HC.representation,
        "image_size": HC.image_size,
    }


@app.post("/reconstruct")
async def reconstruct(file: UploadFile = File(...)):
    if generator is None:
        raise HTTPException(status_code=503, detail="model not loaded (missing models/generator.pt)")

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
        recon = generator(z0, holo)[0, 0]

    return {
        "input":          tensor_to_png_datauri(clean),
        "hologram":       tensor_to_png_datauri(hologram_preview_tensor(field)),
        "reconstruction": tensor_to_png_datauri(recon),
        "meta": {"device": DEVICE, "size": HC.image_size,
                 "representation": HC.representation},
    }


# --------------------------------------------------------------------------- #
#  Agent endpoint (the optical-engineer assistant)
# --------------------------------------------------------------------------- #
class AgentRequest(BaseModel):
    message: str
    model: str = "llama3.2:3b"


@app.post("/agent")
def agent(req: AgentRequest):
    """
    Run the LangGraph optical-engineer agent on a natural-language request.
    Returns the final answer plus the tool-call trajectory for the UI to show.
    """
    from agent_service import run_agent_web
    return run_agent_web(req.message, model=req.model)


# --- serve the frontend ---
FRONTEND_DIR = os.path.join(ROOT, "frontend")


@app.get("/")
def index():
    idx = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"msg": "HoloReconstruct API running. See /docs"}


if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
