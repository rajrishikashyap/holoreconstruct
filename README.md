# HoloReconstruct

Physics-informed holographic image reconstruction — a CVAE-GAN trained with a
differentiable Angular Spectrum Method (ASM) physics loss, served through a
FastAPI backend and a hand-built dashboard.

**Result:** 39.32 dB PSNR / 0.9963 SSIM on 10,000 held-out test images
(+21.29 dB over naive analytic reconstruction).

---

## Folder map

```
holoreconstruct/
├── core/                     the machine-learning core (Phases 1–2.5)
│   ├── propagation_asm.py    ASM physics engine (forward + inverse)
│   ├── dataset.py            image → synthetic hologram data pipeline
│   ├── networks.py           CVAE-GAN: encoder, U-Net generator, PatchGAN
│   ├── train.py              training loop (4-term physics-informed loss)
│   └── evaluate.py           PSNR / SSIM evaluation vs. analytic baseline
├── backend/
│   └── main.py               FastAPI server (loads models/generator.pt)
├── frontend/
│   └── index.html            the dashboard (upload → hologram → reconstruction)
├── models/
│   └── generator.pt          TRAINED WEIGHTS — download from your Google Drive
├── requirements.txt
└── README.md
```

> `models/generator.pt` is the only piece not in this repo by default (it's a
> binary artifact). Copy it from your Drive `holorecon/` folder into `models/`.

---

## Quick start (serve the app)

From the project root, with your conda environment active:

```bash
# 1. install dependencies (see requirements.txt for the correct torch build)
pip install -r requirements.txt

# 2. make sure the trained model is in place
#    models/generator.pt   (copy from Google Drive)

# 3. run the server
uvicorn backend.main:app --reload --port 8000
```

Then open **http://localhost:8000** for the dashboard, or
**http://localhost:8000/docs** for the auto-generated API.

Check it's healthy: visit **http://localhost:8000/health** — you should see
`"model_loaded": true`. If it's `false`, `models/generator.pt` is missing.

---

## Retrain from scratch (optional)

Training runs best on a GPU (Colab T4). From `core/`:

```bash
python train.py            # full run, checkpoints to Drive/local
python evaluate.py         # PSNR/SSIM on the test set
```

All four core files are self-testing — run any of them directly
(`python propagation_asm.py`, etc.) to verify it works.

---

## Notes

- The model was trained on **MNIST**, so it reconstructs digit-like, high-
  contrast shapes best. Complex photos will reconstruct roughly — expected.
- **Demo mode:** the app turns an uploaded image into a *synthetic* hologram
  (via ASM), then reconstructs it. This shows the full pipeline end to end.
- Built entirely on free, open-source tools; runs on consumer hardware.
