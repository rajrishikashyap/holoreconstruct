"""
evaluate.py
===========
Phase 2.5 — Quantitative evaluation for HoloReconstruct.

Turns "the reconstructions look good" into research-grade numbers:

  * PSNR (Peak Signal-to-Noise Ratio, dB) -- higher is better; >30 dB is good,
    >35 dB is very good for this kind of reconstruction.
  * SSIM (Structural Similarity, 0..1) -- higher is better; >0.9 is strong,
    >0.95 is excellent. Correlates with human perception better than PSNR.

It reports two things so your numbers have CONTEXT:
  1. The NEURAL model (generator.pt) on the held-out test set.
  2. A BASELINE: the analytic ASM inverse on the *noisy* hologram -- i.e. what
     you'd get WITHOUT the neural network. This is the number your model must
     beat, and it makes your result meaningful in a writeup.

Usage in Colab:
    !python evaluate.py
    # optional: !python evaluate.py --n 2000   (limit samples for speed)

Requires: propagation_asm.py, dataset.py, networks.py, and a trained
generator.pt in the output folder. Also: pip install scikit-image (Colab has it).
"""

import os
import argparse
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from dataset import HoloConfig, make_mnist_loaders
from networks import Generator
from propagation_asm import propagate


def to_img(t):
    """(1,H,W) or (H,W) tensor -> float32 numpy in [0,1], clipped."""
    a = t.detach().cpu().float().squeeze().numpy()
    return np.clip(a, 0.0, 1.0)


def metrics_pair(true_img, pred_img):
    """PSNR (dB) and SSIM for one image pair in [0,1]."""
    p = sk_psnr(true_img, pred_img, data_range=1.0)
    s = sk_ssim(true_img, pred_img, data_range=1.0)
    # PSNR can be inf for identical images; cap for averaging sanity
    if not np.isfinite(p):
        p = 100.0
    return p, s


def analytic_baseline(holo_batch, hc):
    """
    Baseline reconstruction WITHOUT the network: treat the (normalised, noisy)
    hologram's real+imag as a complex field and back-propagate by -z.
    This is the classical method the neural net is meant to beat.
    Only meaningful for the 'reim' representation.
    """
    if hc.representation != "reim":
        return None
    real = holo_batch[:, 0]
    imag = holo_batch[:, 1]
    field = torch.complex(real, imag)               # (B,H,W)
    back = propagate(field, -hc.z, hc.wavelength, hc.pixel_size)
    amp = back.abs()
    # normalise each to [0,1] for a fair comparison
    B = amp.shape[0]
    out = []
    for i in range(B):
        a = amp[i]
        a = (a - a.min()) / (a.max() - a.min()).clamp_min(1e-8)
        out.append(a)
    return torch.stack(out).unsqueeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0,
                        help="limit number of test images (0 = all)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hc = HoloConfig()
    holo_c = 1 if hc.representation == "intensity" else 2

    folder = ("/content/drive/MyDrive/holorecon"
              if os.path.isdir("/content/drive/MyDrive/holorecon") else "./holorecon")
    gen_path = os.path.join(folder, "generator.pt")
    print(f"loading generator: {gen_path}")

    gen = Generator(holo_c, latent_dim=128, image_size=hc.image_size).to(device)
    gen.load_state_dict(torch.load(gen_path, map_location=device))
    gen.eval()

    _, test_loader = make_mnist_loaders(hc, batch_size=64, num_workers=2)

    psnr_model, ssim_model = [], []
    psnr_base, ssim_base = [], []
    seen = 0

    with torch.no_grad():
        for holo, img in test_loader:
            holo, img = holo.to(device), img.to(device)
            B = img.shape[0]

            z0 = torch.zeros(B, 128, device=device)
            recon = gen(z0, holo)

            base = analytic_baseline(holo, hc)

            for i in range(B):
                t = to_img(img[i])
                p, s = metrics_pair(t, to_img(recon[i]))
                psnr_model.append(p); ssim_model.append(s)
                if base is not None:
                    pb, sb = metrics_pair(t, to_img(base[i]))
                    psnr_base.append(pb); ssim_base.append(sb)

            seen += B
            if args.n and seen >= args.n:
                break

    def stats(xs):
        xs = np.array(xs)
        return xs.mean(), xs.std()

    pm, pms = stats(psnr_model)
    sm, sms = stats(ssim_model)

    print("\n" + "=" * 52)
    print(f"  EVALUATION on {seen} held-out test images")
    print("=" * 52)
    print(f"  NEURAL MODEL (CVAE-GAN + physics)")
    print(f"    PSNR : {pm:6.2f} dB   (+/- {pms:.2f})")
    print(f"    SSIM : {sm:6.4f}      (+/- {sms:.4f})")
    if psnr_base:
        pb, pbs = stats(psnr_base)
        sb, sbs = stats(ssim_base)
        print(f"\n  ANALYTIC BASELINE (no network, ASM back-prop)")
        print(f"    PSNR : {pb:6.2f} dB   (+/- {pbs:.2f})")
        print(f"    SSIM : {sb:6.4f}      (+/- {sbs:.4f})")
        print(f"\n  IMPROVEMENT from the neural model:")
        print(f"    PSNR : +{pm - pb:.2f} dB")
        print(f"    SSIM : +{sm - sb:.4f}")
    print("=" * 52)

    # save a small text report for your writeup
    report = os.path.join(folder, "eval_report.txt")
    with open(report, "w") as f:
        f.write(f"Test images: {seen}\n")
        f.write(f"Model PSNR: {pm:.2f} dB (+/- {pms:.2f})\n")
        f.write(f"Model SSIM: {sm:.4f} (+/- {sms:.4f})\n")
        if psnr_base:
            f.write(f"Baseline PSNR: {pb:.2f} dB (+/- {pbs:.2f})\n")
            f.write(f"Baseline SSIM: {sb:.4f} (+/- {sbs:.4f})\n")
    print(f"saved report -> {report}")


if __name__ == "__main__":
    main()
