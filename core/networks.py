"""
networks.py
===========
Phase 2 — The CVAE-GAN architecture for HoloReconstruct.

Three components (see README section 8 / your PDF "Phase 2"):

  1. Encoder      : (clean image + hologram) -> latent distribution (mu, logvar)
                    Used ONLY during training.
  2. Generator    : (latent z + hologram condition) -> reconstructed image
                    U-Net style with skip connections from the hologram encoder.
                    This is the ONLY piece kept at inference/deployment.
  3. Discriminator: PatchGAN, judges (image | hologram) real-vs-fake on patches,
                    forcing the generator to produce sharp detail.
                    Used ONLY during training.

Design for a 4GB / T4 budget: modest channel counts, 64x64 input, GroupNorm
(stable at small batch sizes, unlike BatchNorm). Everything is sized so a batch
of 16-32 at 64x64 fits comfortably.

Go/no-go gate: if adversarial training is unstable, you stop calling the
Discriminator and train the Generator with pixel+KL(+physics) only. Same
Generator, no rewrite — that IS the VAE fallback.

Requires: nothing beyond torch. Pairs with dataset.py (Phase 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Building blocks
# --------------------------------------------------------------------------- #
def gn(channels, max_groups=8):
    """GroupNorm with a group count that always divides `channels`."""
    g = max_groups
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class ConvBlock(nn.Module):
    """(conv -> GroupNorm -> SiLU) x2, keeps spatial size."""
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), gn(cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), gn(cout), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """ConvBlock then downsample by 2 (returns both for skip connections)."""
    def __init__(self, cin, cout):
        super().__init__()
        self.block = ConvBlock(cin, cout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.block(x)
        return self.pool(skip), skip


class Up(nn.Module):
    """Upsample by 2, concat skip, then ConvBlock."""
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin, 2, stride=2)
        self.block = ConvBlock(cin + cskip, cout)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if off-by-one from odd sizes (safe no-op at powers of two)
        dy, dx = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


# --------------------------------------------------------------------------- #
#  1. Encoder  (image + hologram) -> (mu, logvar)
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    """
    Compresses the (clean image ++ hologram condition) into a latent Gaussian.
    Training only. Input channels = 1 (image) + holo_channels.
    """
    def __init__(self, holo_channels=2, base=32, latent_dim=128, image_size=64):
        super().__init__()
        cin = 1 + holo_channels
        self.d1 = Down(cin, base)          # 64 -> 32
        self.d2 = Down(base, base * 2)     # 32 -> 16
        self.d3 = Down(base * 2, base * 4) # 16 -> 8
        self.bottleneck = ConvBlock(base * 4, base * 4)
        feat = (image_size // 8) ** 2 * base * 4
        self.fc_mu     = nn.Linear(feat, latent_dim)
        self.fc_logvar = nn.Linear(feat, latent_dim)

    def forward(self, image, hologram):
        x = torch.cat([image, hologram], dim=1)
        x, _ = self.d1(x)
        x, _ = self.d2(x)
        x, _ = self.d3(x)
        x = self.bottleneck(x)
        x = torch.flatten(x, 1)
        return self.fc_mu(x), self.fc_logvar(x)


# --------------------------------------------------------------------------- #
#  2. Generator  (latent z + hologram) -> reconstructed image   (U-Net)
# --------------------------------------------------------------------------- #
class Generator(nn.Module):
    """
    U-Net conditioned on the hologram, with the latent z injected at the
    bottleneck. Skip connections come from the HOLOGRAM encoder path, so
    structural detail from the hologram flows straight to the output.

    This is the piece kept at inference: give it a hologram (+ z, or z=0 for a
    deterministic reconstruction) and it outputs the image.
    """
    def __init__(self, holo_channels=2, base=32, latent_dim=128, image_size=64):
        super().__init__()
        self.image_size = image_size
        self.base = base

        # encoder path over the hologram (produces skips)
        self.d1 = Down(holo_channels, base)   # 64 -> 32
        self.d2 = Down(base, base * 2)         # 32 -> 16
        self.d3 = Down(base * 2, base * 4)     # 16 -> 8
        self.bottleneck = ConvBlock(base * 4, base * 4)

        # project latent z and add it into the bottleneck
        self.bott_hw = image_size // 8
        self.fc_z = nn.Linear(latent_dim, base * 4 * self.bott_hw * self.bott_hw)

        # decoder path (skips are the pre-pool features from d1/d2/d3)
        self.u3 = Up(base * 4, base * 4, base * 2)  # 8 -> 16
        self.u2 = Up(base * 2, base * 2, base)      # 16 -> 32
        self.u1 = Up(base, base, base)              # 32 -> 64
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, z, hologram):
        x, s1 = self.d1(hologram)
        x, s2 = self.d2(x)
        x, s3 = self.d3(x)
        x = self.bottleneck(x)

        # inject latent at bottleneck
        zt = self.fc_z(z).view(-1, self.base * 4, self.bott_hw, self.bott_hw)
        x = x + zt

        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        return torch.sigmoid(self.out(x))   # image in [0,1]


# --------------------------------------------------------------------------- #
#  3. Discriminator  (image | hologram) -> patch real/fake map   (PatchGAN)
# --------------------------------------------------------------------------- #
class Discriminator(nn.Module):
    """
    PatchGAN: outputs a grid of real/fake scores (one per receptive-field
    patch) instead of a single number, which pushes local sharpness. Sees the
    image concatenated with its hologram condition (conditional GAN).
    Training only.
    """
    def __init__(self, holo_channels=2, base=32):
        super().__init__()
        cin = 1 + holo_channels

        def block(ci, co, stride=2, norm=True):
            layers = [nn.Conv2d(ci, co, 4, stride=stride, padding=1)]
            if norm:
                layers.append(gn(co))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(cin, base, norm=False),   # 64 -> 32
            *block(base, base * 2),          # 32 -> 16
            *block(base * 2, base * 4),      # 16 -> 8
            nn.Conv2d(base * 4, 1, 4, padding=1),  # -> patch score map
        )

    def forward(self, image, hologram):
        return self.net(torch.cat([image, hologram], dim=1))


# --------------------------------------------------------------------------- #
#  Reparameterization helper (the VAE sampling trick)
# --------------------------------------------------------------------------- #
def reparameterize(mu, logvar):
    """Sample z = mu + sigma * eps, so gradients flow through mu/logvar."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# --------------------------------------------------------------------------- #
#  Self-test: run `python networks.py`
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing CVAE-GAN on: {device}")

    B, holo_c, S, L = 4, 2, 64, 128
    enc = Encoder(holo_c, latent_dim=L, image_size=S).to(device)
    gen = Generator(holo_c, latent_dim=L, image_size=S).to(device)
    dis = Discriminator(holo_c).to(device)

    image = torch.rand(B, 1, S, S, device=device)
    holo  = torch.randn(B, holo_c, S, S, device=device)

    # forward through the whole training path
    mu, logvar = enc(image, holo)
    z = reparameterize(mu, logvar)
    recon = gen(z, holo)
    dscore = dis(recon, holo)

    print(f"latent mu      : {tuple(mu.shape)}      (expect ({B}, {L}))")
    print(f"reconstruction : {tuple(recon.shape)}  (expect ({B}, 1, {S}, {S}))")
    print(f"recon range    : [{recon.min():.2f}, {recon.max():.2f}]  (expect [0,1])")
    print(f"disc patch map : {tuple(dscore.shape)}   (patch grid, not a scalar)")

    # inference path: no encoder, z = 0 (deterministic)
    z0 = torch.zeros(B, L, device=device)
    infer = gen(z0, holo)
    print(f"inference out  : {tuple(infer.shape)}  (generator alone works)")

    # param counts (deployment cares about generator size)
    def count(m): return sum(p.numel() for p in m.parameters()) / 1e6
    print(f"\nparams (M): encoder={count(enc):.2f}  generator={count(gen):.2f}  "
          f"discriminator={count(dis):.2f}")
    print(f"kept at inference: generator only = {count(gen):.2f}M params")

    ok = (mu.shape == (B, L) and recon.shape == (B, 1, S, S)
          and 0 <= recon.min() and recon.max() <= 1)
    print("PASS" if ok else "CHECK: shapes/ranges unexpected")
