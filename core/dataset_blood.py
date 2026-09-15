"""
dataset_blood.py
================
Cell-microscopy data pipeline for HoloReconstruct (Phase: harder dataset).

Same job as dataset.py, but the source images are REAL blood-cell microscopy
(BloodMNIST, part of MedMNIST): 17,092 images of normal blood cells across 8
classes (neutrophils, eosinophils, basophils, lymphocytes, monocytes, immature
granulocytes, erythroblasts, platelets). This makes the project genuinely about
medical imaging.

Each image is turned into a synthetic hologram on the fly via the SAME verified
ASM forward model, then reconstructed by the network -- identical pipeline,
harder data. The loader exposes make_blood_loaders(cfg, ...) with the same
(hologram, clean_image) output shape as MNIST, so train.py and evaluate.py work
unchanged.

Two differences from MNIST, both handled here:
  * BloodMNIST is COLOR (3-channel). Holography works on intensity, so we
    convert to grayscale -- physically standard and keeps the pipeline 1-channel.
  * It ships at native 28/64/128/224 px; we request size=64 to match image_size.

Install once:  pip install medmnist
First run downloads ~30 MB to ~/.medmnist (free).

Requires: propagation_asm.py (Phase 1) in the same folder / on the path.
"""

import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from medmnist import BloodMNIST

from propagation_asm import image_to_hologram
# reuse the exact hologram-building helpers from the MNIST pipeline
from dataset import (HoloConfig, _complex_to_channels,
                     _add_sensor_noise, _normalise_per_sample)


class BloodHologramDataset(Dataset):
    """
    Wraps a BloodMNIST split and produces (hologram, clean_image) pairs,
    identical in shape to the MNIST HologramDataset.

    Returns:
        hologram : FloatTensor (C, H, W)  -- C = 1 or 2 per representation
        image    : FloatTensor (1, H, W)  -- grayscale cell image in [0,1]
    """

    def __init__(self, split, cfg: HoloConfig, root=None):
        # size=cfg.image_size gives the native 64x64 (or 128) version directly.
        # as_rgb=False + our grayscale conversion keeps it single-channel.
        # medmnist needs an existing root dir; default to ./medmnist_data and
        # create it (its own default ~/.medmnist isn't auto-made on Colab).
        if root is None:
            root = os.path.join(os.getcwd(), "medmnist_data")
        os.makedirs(root, exist_ok=True)
        tfm = transforms.Compose([transforms.ToTensor()])  # -> (C,H,W) in [0,1]
        self.base = BloodMNIST(
            split=split, transform=tfm, download=True,
            size=cfg.image_size, root=root,
        )
        self.cfg = cfg

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        cfg = self.cfg
        img, _label = self.base[idx]          # (3, H, W) float in [0,1]

        # 1. to grayscale (mean over channels) -> (H, W)
        if img.dim() == 3 and img.shape[0] > 1:
            gray = img.mean(dim=0)
        else:
            gray = img[0]
        gray = gray.clamp(0, 1)
        clean = gray.unsqueeze(0)             # (1, H, W) target

        # 2. forward physics: image -> complex hologram
        field = image_to_hologram(gray, cfg.z, cfg.wavelength, cfg.pixel_size)

        # 3. complex -> channels, 4. noise, 5. normalise  (same as MNIST)
        holo = _complex_to_channels(field, cfg.representation)
        if cfg.add_noise:
            holo = _add_sensor_noise(holo, cfg)
        holo = _normalise_per_sample(holo)

        return holo.float(), clean.float()


def make_blood_loaders(cfg: HoloConfig, batch_size=32, root=None, num_workers=2):
    """
    Build train/test DataLoaders of (hologram, clean_cell_image) pairs from
    BloodMNIST. Same signature/behaviour as make_mnist_loaders so train.py and
    evaluate.py can use it by swapping one import.
    """
    train_ds = BloodHologramDataset("train", cfg, root=root)
    test_ds  = BloodHologramDataset("test",  cfg, root=root)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# --------------------------------------------------------------------------- #
#  self-test: run `python dataset_blood.py`
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = HoloConfig()
    print(f"representation = {cfg.representation} | size = {cfg.image_size} | "
          f"add_noise = {cfg.add_noise}")

    train_loader, test_loader = make_blood_loaders(cfg, batch_size=8)
    holo, img = next(iter(train_loader))

    exp_c = 1 if cfg.representation == "intensity" else 2
    print(f"hologram batch : {tuple(holo.shape)}   (expect (8, {exp_c}, "
          f"{cfg.image_size}, {cfg.image_size}))")
    print(f"image batch    : {tuple(img.shape)}   (expect (8, 1, "
          f"{cfg.image_size}, {cfg.image_size}))")
    print(f"image range    : [{img.min():.2f}, {img.max():.2f}]   (expect ~[0,1])")
    print(f"train size     : {len(train_loader.dataset)} cells")
    print(f"test size      : {len(test_loader.dataset)} cells")
    ok = (holo.shape == (8, exp_c, cfg.image_size, cfg.image_size)
          and img.shape == (8, 1, cfg.image_size, cfg.image_size))
    print("PASS" if ok else "CHECK: shapes unexpected")
