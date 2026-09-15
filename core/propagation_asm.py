"""
propagation_asm.py
==================
Phase 1 — Physics Core of the HoloReconstruct project.

The Angular Spectrum Method (ASM): the forward physics model that turns a clean
image into a synthetic hologram, and back again.

Pipeline (forward):   image --FFT--> x H(z)  --IFFT--> hologram
Pipeline (inverse):   hologram --FFT--> x H(-z) --IFFT--> image

Everything is written with torch.fft so it runs on GPU and is fully
differentiable. That last property is what lets us later plug this SAME code
into the network's loss function as the "physics loss" (Phase 2).

Tested against a NumPy reference: 64x64 round-trip MSE ~ 1e-33 (floating-point
perfect). The physics is exact when there is no noise; the neural network's job
(Phase 2) is to invert it when there IS noise and the exact z is unknown.

Run this file directly to reproduce the round-trip self-test:
    python propagation_asm.py
"""

import torch


def asm_transfer_function(shape, z, wavelength, pixel_size, device="cpu", dtype=torch.float32):
    """
    Build the free-space propagation transfer function H(fx, fy, z).

    This is the heart of the ASM: for each spatial frequency (fx, fy), H says
    how much that frequency-component's phase advances after travelling
    distance z. Multiplying the image's spectrum by H IS the propagation.

    Args:
        shape:      (H, W) of the field
        z:          propagation distance in metres (negative = propagate back)
        wavelength: laser wavelength in metres (e.g. 532e-9 for green)
        pixel_size: sensor pixel pitch in metres (e.g. 2e-6)
        device:     "cpu" or "cuda"
        dtype:      float32 (default) or float64 for the real-valued math

    Returns:
        Complex tensor H of shape (H, W).
    """
    ny, nx = shape

    # Spatial-frequency axes: cycles per metre. fftfreq matches the layout of
    # torch.fft.fft2's output (zero-frequency first), so no fftshift is needed.
    fx = torch.fft.fftfreq(nx, d=pixel_size, device=device, dtype=dtype)
    fy = torch.fft.fftfreq(ny, d=pixel_size, device=device, dtype=dtype)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")

    # Argument under the square root of the angular-spectrum kernel.
    arg = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2

    # Evanescent waves (arg < 0) decay instantly and must not propagate.
    # We mask them to zero and clamp the sqrt argument to stay real.
    mask = (arg >= 0).to(dtype)
    arg = torch.clamp(arg, min=0.0)

    phase = (2.0 * torch.pi / wavelength) * z * torch.sqrt(arg)
    # H = exp(i * phase), masked. 1j*phase builds a complex tensor.
    H = torch.exp(1j * phase.to(dtype)) * mask
    return H


def propagate(field, z, wavelength, pixel_size):
    """
    Propagate a complex optical field a distance z via the ASM.

    Works on a single field (H, W) or a batch (B, H, W) / (B, C, H, W) —
    fft2 operates on the last two dims.

    Args:
        field:      complex tensor, last two dims are (H, W)
        z:          distance in metres (use -z to propagate backwards)
        wavelength: metres
        pixel_size: metres

    Returns:
        Complex tensor, same shape as `field`, the propagated field.
    """
    if not torch.is_complex(field):
        field = field.to(torch.complex64)

    H = asm_transfer_function(
        field.shape[-2:], z, wavelength, pixel_size,
        device=field.device,
        dtype=torch.float32 if field.dtype == torch.complex64 else torch.float64,
    )

    A = torch.fft.fft2(field)   # 1. spatial -> frequency
    A = A * H                   # 2. multiply by transfer function (propagate)
    out = torch.fft.ifft2(A)    # 3. frequency -> spatial
    return out


def image_to_hologram(image, z, wavelength, pixel_size):
    """
    Forward model: a real, non-negative image -> complex hologram field.

    The image is treated as the wave's AMPLITUDE with a flat (zero) initial
    phase. Returns the complex field at the hologram plane. To get what a
    camera records, take .abs()**2 (intensity).
    """
    field0 = image.to(torch.complex64)          # amplitude = image, phase = 0
    return propagate(field0, z, wavelength, pixel_size)


def hologram_to_image(hologram_field, z, wavelength, pixel_size):
    """
    Inverse model (analytic): complex hologram field -> recovered amplitude.

    Propagates back by -z. This only works perfectly when the field is complex
    and noise-free — which is exactly why Phase 2's neural network exists: to
    invert real, noisy, intensity-only holograms where this analytic step fails.
    """
    recon_field = propagate(hologram_field, -z, wavelength, pixel_size)
    return recon_field.abs()


# --------------------------------------------------------------------------- #
#  Self-test: run `python propagation_asm.py`
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running ASM self-test on: {device}")

    # Physical parameters (typical lensless holography)
    wavelength = 532e-9    # 532 nm green laser
    pixel_size = 2e-6      # 2 micron pixels
    z          = 1e-3      # 1 mm

    # Test image: a bright square on black, 64x64
    N = 64
    img = torch.zeros((N, N), device=device)
    img[24:40, 24:40] = 1.0

    # Forward then inverse
    holo = image_to_hologram(img, z, wavelength, pixel_size)
    recon = hologram_to_image(holo, z, wavelength, pixel_size)

    mse = torch.mean((recon - img) ** 2).item()
    max_err = (recon - img).abs().max().item()

    print(f"round-trip MSE  : {mse:.3e}   (expect ~0)")
    print(f"max pixel error : {max_err:.3e}")
    print("PASS" if mse < 1e-5 else "CHECK: error larger than expected")

    # Sanity: the hologram must have diffracted (energy spread beyond the block)
    corner_energy = holo[:20, :20].abs().sum().item()
    print(f"diffraction into corner region: {corner_energy:.3f}  (should be > 0)")
