# Models

Trained model weights are **not** committed to this repo (binary artifacts, kept
out per standard practice). You have two options to get a `generator.pt` here:

# Option A - retrain (recommended, fast & free)

Training is quick on a free Colab T4 GPU (~15 min for MNIST, ~30 min for cells):

```bash
# MNIST digits (quick sanity model)
python core/train.py

# Blood-cell microscopy (the medical result)
python core/train_blood.py
```

Each run saves `generator.pt` (the ~2 MB deployment model) to its output folder.
Copy the one you want into this `models/` directory as `generator.pt`.

# Option B - download pretrained weights

If a release is attached to this repo, download `generator.pt` from the
**Releases** page and place it here as `models/generator.pt`.

# Which model does the app serve?

`backend/main.py` loads `models/generator.pt`. Drop in the MNIST model for the
digit demo, or the blood-cell model for the medical demo - the app auto-detects
device (CPU/GPU) and serves whichever is present.
