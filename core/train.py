"""
train.py
========
Phase 2 — Training loop for the HoloReconstruct CVAE-GAN.

Ties together:
  * dataset.py          -> (hologram, clean_image) pairs
  * networks.py         -> Encoder, Generator (U-Net), Discriminator (PatchGAN)
  * propagation_asm.py  -> differentiable ASM, reused as the PHYSICS LOSS

The four-term generator loss (README section 7):

    L_total = a*L_pixel + b*L_KLD + g*L_adversarial + d*L_physics

  L_pixel       : L1 between reconstruction and clean image (geometry)
  L_KLD         : keeps the VAE latent ~ N(0, I) (smooth latent)
  L_adversarial : fool the PatchGAN discriminator (sharpness)
  L_physics     : re-propagate the reconstruction through ASM and match the
                  input hologram (optical fidelity -- the differentiator)

Features:
  * checkpointing to Google Drive (survives Colab disconnects)
  * TensorBoard logging (watch G vs D loss -- GAN stability)
  * periodic sample-image grids (see it learn)
  * GO/NO-GO GATE: set USE_GAN=False to drop the discriminator and train a
    plain conditional VAE (+physics). Same generator, the built-in fallback.

Usage in Colab:
    # (optional) mount Drive first so checkpoints persist:
    from google.colab import drive; drive.mount('/content/drive')
    !python train.py            # full run
    !python train.py --smoke    # 30-step smoke test: confirms nothing crashes
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from dataset import HoloConfig, make_mnist_loaders
from networks import Encoder, Generator, Discriminator, reparameterize
from propagation_asm import image_to_hologram


# --------------------------------------------------------------------------- #
#  Training configuration
# --------------------------------------------------------------------------- #
class TrainConfig:
    # loss weights (a, b, g, d in the equation)
    w_pixel       = 1.0
    w_kld         = 0.001    # small: KL easily dominates and blurs if too high
    w_adversarial = 0.01     # small: adversarial term is a nudge, not the boss
    w_physics     = 0.1      # the physics leash

    # optimisation
    lr_g = 2e-4
    lr_d = 2e-4
    betas = (0.5, 0.999)     # standard GAN Adam betas
    batch_size = 32
    epochs = 20
    latent_dim = 128

    # gate: flip to False to train VAE-only (no discriminator)
    use_gan = True

    # io -- points at Drive if mounted, else local
    @property
    def out_dir(self):
        drive = "/content/drive/MyDrive/holorecon"
        base = drive if os.path.isdir("/content/drive/MyDrive") else "./holorecon"
        os.makedirs(base, exist_ok=True)
        return base

    log_every = 50
    sample_every = 500
    ckpt_every_epochs = 1


def kld_loss(mu, logvar):
    """KL divergence between N(mu, sigma) and N(0, I), mean over batch."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def physics_loss(recon, input_holo, holo_cfg):
    """
    Re-propagate the reconstruction through ASM and compare to the input
    hologram, in the SAME normalised space the dataset used.

    recon      : (B, 1, H, W) in [0,1]
    input_holo : (B, C, H, W) the (already-normalised) network input hologram
    """
    B = recon.shape[0]
    reprop_channels = []
    for i in range(B):
        # forward physics on the reconstruction (amplitude=recon, phase=0)
        field = image_to_hologram(
            recon[i, 0], holo_cfg.z, holo_cfg.wavelength, holo_cfg.pixel_size
        )
        if holo_cfg.representation == "reim":
            ch = torch.stack([field.real, field.imag], dim=0)
        elif holo_cfg.representation == "ampphase":
            ch = torch.stack([field.abs(), field.angle()], dim=0)
        else:  # intensity
            ch = (field.abs() ** 2).unsqueeze(0)
        # normalise identically to dataset._normalise_per_sample
        ch = (ch - ch.mean()) / ch.std().clamp_min(1e-8)
        reprop_channels.append(ch)
    reprop = torch.stack(reprop_channels, dim=0)
    return F.l1_loss(reprop, input_holo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="run ~30 steps to confirm the loop works, then exit")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc = TrainConfig()
    hc = HoloConfig()
    print(f"device={device}  use_gan={tc.use_gan}  representation={hc.representation}")
    print(f"outputs -> {tc.out_dir}")

    # TensorBoard (optional import so the file runs even without it)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(tc.out_dir, "runs"))
    except Exception as e:
        print(f"(TensorBoard unavailable: {e})")

    # data
    holo_channels = 1 if hc.representation == "intensity" else 2
    train_loader, test_loader = make_mnist_loaders(
        hc, batch_size=tc.batch_size, num_workers=2)

    # models
    enc = Encoder(holo_channels, latent_dim=tc.latent_dim, image_size=hc.image_size).to(device)
    gen = Generator(holo_channels, latent_dim=tc.latent_dim, image_size=hc.image_size).to(device)
    dis = Discriminator(holo_channels).to(device) if tc.use_gan else None

    # optimisers: generator+encoder share one, discriminator its own
    opt_g = torch.optim.Adam(
        list(enc.parameters()) + list(gen.parameters()), lr=tc.lr_g, betas=tc.betas)
    opt_d = (torch.optim.Adam(dis.parameters(), lr=tc.lr_d, betas=tc.betas)
             if tc.use_gan else None)

    bce = nn.BCEWithLogitsLoss()
    fixed_holo, fixed_img = next(iter(test_loader))
    fixed_holo, fixed_img = fixed_holo[:16].to(device), fixed_img[:16].to(device)

    step = 0
    for epoch in range(tc.epochs):
        for holo, img in train_loader:
            holo, img = holo.to(device), img.to(device)
            B = img.shape[0]

            # ---------------- Generator + Encoder ----------------
            opt_g.zero_grad()
            mu, logvar = enc(img, holo)
            z = reparameterize(mu, logvar)
            recon = gen(z, holo)

            l_pixel = F.l1_loss(recon, img)
            l_kld = kld_loss(mu, logvar)
            l_phys = physics_loss(recon, holo, hc)

            l_adv = torch.tensor(0.0, device=device)
            if tc.use_gan:
                # generator wants disc to call its recon "real" (label 1)
                d_fake = dis(recon, holo)
                l_adv = bce(d_fake, torch.ones_like(d_fake))

            l_g = (tc.w_pixel * l_pixel + tc.w_kld * l_kld
                   + tc.w_adversarial * l_adv + tc.w_physics * l_phys)
            l_g.backward()
            opt_g.step()

            # ---------------- Discriminator ----------------
            l_d = torch.tensor(0.0, device=device)
            if tc.use_gan:
                opt_d.zero_grad()
                d_real = dis(img, holo)
                d_fake = dis(recon.detach(), holo)   # detach: don't train G here
                l_d = 0.5 * (bce(d_real, torch.ones_like(d_real))
                             + bce(d_fake, torch.zeros_like(d_fake)))
                l_d.backward()
                opt_d.step()

            # ---------------- logging ----------------
            if step % tc.log_every == 0:
                msg = (f"e{epoch} s{step} | G {l_g.item():.3f} "
                       f"(pix {l_pixel.item():.3f} kld {l_kld.item():.3f} "
                       f"phys {l_phys.item():.3f} adv {l_adv.item():.3f})")
                if tc.use_gan:
                    msg += f" | D {l_d.item():.3f}"
                print(msg)
                if writer:
                    writer.add_scalar("G/total", l_g.item(), step)
                    writer.add_scalar("G/pixel", l_pixel.item(), step)
                    writer.add_scalar("G/kld", l_kld.item(), step)
                    writer.add_scalar("G/physics", l_phys.item(), step)
                    if tc.use_gan:
                        writer.add_scalar("G/adversarial", l_adv.item(), step)
                        writer.add_scalar("D/total", l_d.item(), step)

            # ---------------- sample grid ----------------
            if step % tc.sample_every == 0:
                gen.eval()
                with torch.no_grad():
                    z0 = torch.zeros(fixed_img.shape[0], tc.latent_dim, device=device)
                    samp = gen(z0, fixed_holo)
                grid = make_grid(torch.cat([fixed_img, samp], dim=0),
                                 nrow=fixed_img.shape[0])
                save_image(grid, os.path.join(tc.out_dir, f"sample_{step:06d}.png"))
                if writer:
                    writer.add_image("recon(top=true,bottom=pred)", grid, step)
                gen.train()

            step += 1
            if args.smoke and step >= 30:
                print("SMOKE PASS — loop ran 30 steps, all losses finite:",
                      all(torch.isfinite(t).all() for t in
                          [l_g, l_pixel, l_kld, l_phys, l_adv]))
                return

        # ---------------- checkpoint ----------------
        if (epoch + 1) % tc.ckpt_every_epochs == 0:
            ckpt = {
                "epoch": epoch, "step": step,
                "encoder": enc.state_dict(),
                "generator": gen.state_dict(),
                "discriminator": dis.state_dict() if tc.use_gan else None,
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict() if tc.use_gan else None,
                "train_cfg": {k: v for k, v in vars(TrainConfig).items()
                              if isinstance(v, (int, float, bool, str))},
            }
            path = os.path.join(tc.out_dir, "checkpoint.pt")
            torch.save(ckpt, path)
            # also save a deploy-only generator (the sole inference artifact)
            torch.save(gen.state_dict(), os.path.join(tc.out_dir, "generator.pt"))
            print(f"  saved checkpoint + generator.pt @ epoch {epoch}")

    if writer:
        writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
