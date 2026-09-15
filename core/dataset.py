"""
dataset.py
==========
Phase 2 — Data pipeline for the HoloReconstruct project.

Takes clean images (MNIST to start) and, on the fly, runs each one through the
verified ASM forward model from Phase 1 to produce a synthetic hologram. Each
sample the network sees is a pair:

        (hologram_tensor, clean_image_tensor)

The clean image is the "answer key"; the hologram is the "question".

Key design choices (both are single config flags so you can experiment):

  * HOLOGRAM REPRESENTATION  -- how the complex hologram is fed to the CNN:
        "reim"      -> 2 channels: [real, imaginary]   (RECOMMENDED default)
        "ampphase"  -> 2 channels: [amplitude, phase]
        "intensity" -> 1 channel:  |field|^2 (what a real camera sees; hard mode)

  * NOISE  -- adds sensor noise so the network learns to denoise:
        Gaussian (read noise) + optional Poisson (shot noise).

Nothing here needs a GPU; holograms are generated per-sample on CPU in the
DataLoader workers, which keeps VRAM free for the model. torch.fft is fast
enough at 64x64 that this is not a bottleneck.

Requires: propagation_asm.py (Phase 1) in the same folder.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from propagation_asm import image_to_hologram


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
class HoloConfig:
    """All physics + pipeline knobs in one place."""
    # optics (must match propagation_asm conventions: metres)
    wavelength = 532e-9     # 532 nm green laser
    pixel_size = 2e-6       # 2 micron sensor pixels
    z          = 1e-3       # 1 mm propagation distance

    # image
    image_size = 64         # resize everything to 64x64 (fits T4 comfortably)

    # hologram representation: "reim" | "ampphase" | "intensity"
    representation = "reim"

    # noise
    add_noise      = True
    gaussian_std   = 0.05   # read noise, relative to normalised hologram
    poisson_photons = 0     # 0 = off; e.g. 1000 = shot noise at ~1000 photons


def _complex_to_channels(field, representation):
    """Convert a complex (H, W) field into the chosen real-valued channels."""
    if representation == "reim":
        return torch.stack([field.real, field.imag], dim=0)        # (2, H, W)
    elif representation == "ampphase":
        return torch.stack([field.abs(), field.angle()], dim=0)    # (2, H, W)
    elif representation == "intensity":
        return (field.abs() ** 2).unsqueeze(0)                     # (1, H, W)
    else:
        raise ValueError(f"unknown representation: {representation}")


def _add_sensor_noise(holo_channels, cfg):
    """Add Poisson (shot) then Gaussian (read) noise to the hologram channels."""
    x = holo_channels
    if cfg.poisson_photons and cfg.poisson_photons > 0:
        # scale to a photon count, sample Poisson, scale back
        scaled = torch.clamp(x, min=0) * cfg.poisson_photons
        x = torch.poisson(scaled) / cfg.poisson_photons
    if cfg.gaussian_std and cfg.gaussian_std > 0:
        x = x + torch.randn_like(x) * cfg.gaussian_std
    return x


def _normalise_per_sample(x):
    """Zero-mean/unit-ish scale per sample so channels are network-friendly."""
    mean = x.mean()
    std = x.std().clamp_min(1e-8)
    return (x - mean) / std


class HologramDataset(Dataset):
    """
    Wraps a base image dataset (e.g. MNIST) and produces (hologram, image) pairs.

    Returns:
        hologram : FloatTensor (C, H, W)  -- C is 1 or 2 per representation
        image    : FloatTensor (1, H, W)  -- the clean target, in [0, 1]
    """

    def __init__(self, base_dataset, cfg: HoloConfig):
        self.base = base_dataset
        self.cfg = cfg

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        cfg = self.cfg

        # 1. get a clean image in [0,1], shape (1, H, W)
        img, _label = self.base[idx]          # torchvision returns (tensor, label)
        if img.dim() == 3 and img.shape[0] > 1:
            img = img.mean(dim=0, keepdim=True)   # force grayscale
        img = img.clamp(0, 1)
        clean = img[0]                            # (H, W)

        # 2. forward physics: image -> complex hologram field
        field = image_to_hologram(
            clean, cfg.z, cfg.wavelength, cfg.pixel_size
        )                                          # complex (H, W)

        # 3. complex field -> real channels per chosen representation
        holo = _complex_to_channels(field, cfg.representation)

        # 4. add sensor noise (the reason the network must learn to denoise)
        if cfg.add_noise:
            holo = _add_sensor_noise(holo, cfg)

        # 5. normalise the hologram channels for stable training
        holo = _normalise_per_sample(holo)

        return holo.float(), img.float()


def make_mnist_loaders(cfg: HoloConfig, batch_size=32, root="./data", num_workers=2):
    """
    Build train/test DataLoaders of (hologram, clean_image) pairs from MNIST.
    Downloads MNIST on first run (~10 MB, free).
    """
    tfm = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),                    # -> (1, H, W) in [0,1]
    ])
    train_base = datasets.MNIST(root, train=True,  download=True, transform=tfm)
    test_base  = datasets.MNIST(root, train=False, download=True, transform=tfm)

    train_ds = HologramDataset(train_base, cfg)
    test_ds  = HologramDataset(test_base, cfg)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# --------------------------------------------------------------------------- #
#  Self-test: run `python dataset.py`
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = HoloConfig()
    print(f"representation = {cfg.representation} | add_noise = {cfg.add_noise}")

    train_loader, test_loader = make_mnist_loaders(cfg, batch_size=8)
    holo, img = next(iter(train_loader))

    exp_c = 1 if cfg.representation == "intensity" else 2
    print(f"hologram batch : {tuple(holo.shape)}   (expect (8, {exp_c}, "
          f"{cfg.image_size}, {cfg.image_size}))")
    print(f"image batch    : {tuple(img.shape)}   (expect (8, 1, "
          f"{cfg.image_size}, {cfg.image_size}))")
    print(f"hologram range : [{holo.min():.2f}, {holo.max():.2f}]  (normalised)")
    print(f"image range    : [{img.min():.2f}, {img.max():.2f}]   (expect ~[0,1])")

    ok = (holo.shape == (8, exp_c, cfg.image_size, cfg.image_size)
          and img.shape == (8, 1, cfg.image_size, cfg.image_size))
    print("PASS" if ok else "CHECK: shapes unexpected")
