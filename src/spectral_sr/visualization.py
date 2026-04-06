"""Spectral visualization utilities – generates publication-quality plots and uploads to R2.

Replicates the visual outputs produced in the original notebook:
  - Spectral reconstruction curves (true / reconstructed / OLI input)
  - Per-band reflectance histograms
  - False-color composite images (Hyperion and reconstructed)

All functions return the PNG bytes so callers can decide whether to save to disk,
upload to R2, or embed in an API response.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless backend – no display required

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wavelength helpers
# ---------------------------------------------------------------------------

def load_hyperion_wavelengths(properties_path: str | Path) -> np.ndarray:
    """Parse a Hyperion properties text file and return wavelengths in µm."""
    import re
    import pandas as pd

    regex = re.compile(r"(\d+) = (.*)")
    bands, waves = [], []

    with open(properties_path, "r") as f:
        for line in f:
            if "Wavelengths" in line:
                m = regex.findall(line)
                if m:
                    bands.append(int(m[0][0]))
                    waves.append(float(m[0][1]) / 1000)  # nm -> µm

    # Drop known bad bands (same mask as in the original code)
    bad = set(range(1, 8)) | set(range(58, 77)) | set(range(225, 243)) \
        | set(range(121, 127)) | set(range(167, 181)) | set(range(222, 242))
    result = [(b, w) for b, w in zip(bands, waves) if b not in bad]
    return np.array([w for _, w in result])  # shape (175,)


def load_landsat_wavelengths(properties_path: str | Path) -> np.ndarray:
    """Parse a Landsat properties text file and return wavelengths in µm."""
    import re

    regex = re.compile(r"(\d+) = (.*)")
    waves = []

    with open(properties_path, "r") as f:
        for line in f:
            if "Wavelengths" in line:
                m = regex.findall(line)
                if m:
                    waves.append(float(m[0][1]))

    return np.array(waves)


# ---------------------------------------------------------------------------
# Core plot generation
# ---------------------------------------------------------------------------

def spectral_curve_plot(
    patch_hyp: np.ndarray,
    patch_landsat: np.ndarray,
    patch_recon: np.ndarray,
    wav_hyp: np.ndarray,
    wav_landsat: np.ndarray,
    row: int,
    col: int,
    scene_name: str = "patch",
) -> bytes:
    """
    Generate a spectral curve comparison plot (PNG bytes).

    Args:
        patch_hyp:     Hyperion patch  (H, W, 175)
        patch_landsat: Landsat patch   (H, W, 7)
        patch_recon:   Reconstructed   (H, W, 175)
        wav_hyp:       Hyperion wavelengths in µm (175,)
        wav_landsat:   Landsat wavelengths in µm  (7,)
        row, col:      Pixel position to plot
        scene_name:    Title label

    Returns:
        PNG bytes
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Spectral Reconstruction – {scene_name}", fontsize=14, fontweight="bold")

    ax.plot(wav_hyp, patch_hyp[row, col, :] / 10000, "o-",
            label="True Hyperion Spectra", markersize=3)
    ax.plot(wav_hyp, patch_recon[row, col, :] / 10000, "o--",
            label="Reconstructed Spectra", markersize=3)
    ax.plot(wav_landsat, patch_landsat[row, col, :] / 10000, "s",
            label="Landsat-8 OLI Input", markersize=6)

    ax.set_xlabel("Wavelength [µm]", fontsize=11)
    ax.set_ylabel("Reflectance (proportion)", fontsize=11)
    ax.legend(fontsize=10)
    sns.despine()

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def band_distribution_plot(
    patch_hyp: np.ndarray,
    patch_recon: np.ndarray,
    band_indices: tuple = (29, 20, 12),
    scene_name: str = "patch",
) -> bytes:
    """
    Plot reflectance distributions for selected bands (ground truth vs reconstructed).

    Args:
        patch_hyp:    Hyperion ground truth  (H, W, 175)
        patch_recon:  Reconstructed          (H, W, 175)
        band_indices: Bands to plot
        scene_name:   Title label

    Returns:
        PNG bytes
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Band Distribution – {scene_name}", fontsize=14, fontweight="bold")

    for b in band_indices:
        axes[0].set_title("Ground Truth (Hyperion)")
        sns.histplot(patch_hyp[:, :, b].flatten() / 10000, ax=axes[0],
                     kde=True, label=f"Band {b}")

    for b in band_indices:
        axes[1].set_title("Reconstructed")
        sns.histplot(patch_recon[:, :, b].flatten() / 10000, ax=axes[1],
                     kde=True, label=f"Band {b}")

    for ax in axes:
        ax.set_xlabel("Reflectance (proportion)")
        ax.legend()
    sns.despine()

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def false_color_composite_plot(
    patch_hyp: np.ndarray,
    patch_recon: np.ndarray,
    rgb_bands: tuple = (29, 20, 12),
    row: int = None,
    col: int = None,
    scene_name: str = "patch",
) -> bytes:
    """
    Side-by-side false-color composite images (ground truth and reconstruction).

    Args:
        patch_hyp:  Hyperion ground truth  (H, W, 175)
        patch_recon: Reconstructed         (H, W, 175)
        rgb_bands:  Three band indices to use as RGB channels
        row, col:   Optional crosshair position
        scene_name: Title label

    Returns:
        PNG bytes
    """
    def _normalize(arr):
        arr = arr.astype(np.float32)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        return np.clip(arr, 0, 1)

    r, g, b = rgb_bands

    gt_rgb = _normalize(np.stack([
        patch_hyp[:, :, r],
        patch_hyp[:, :, g],
        patch_hyp[:, :, b],
    ], axis=-1))
    rc_rgb = _normalize(np.stack([
        patch_recon[:, :, r],
        patch_recon[:, :, g],
        patch_recon[:, :, b],
    ], axis=-1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"False-Color Composite (bands {rgb_bands}) – {scene_name}",
                 fontsize=13, fontweight="bold")

    for ax, img, title in zip(axes, [gt_rgb, rc_rgb],
                               ["Ground Truth (Hyperion)", "Reconstructed"]):
        ax.imshow(img)
        ax.set_title(title)
        if row is not None and col is not None:
            ax.axvline(col, color="red", linewidth=1)
            ax.axhline(row, color="red", linewidth=1)
        ax.axis("off")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_reconstruction_plots(
    model,
    landsat_dir: str | Path,
    hyperion_dir: str | Path,
    output_dir: str | Path,
    wav_hyp: np.ndarray,
    wav_landsat: np.ndarray,
    num_samples: int = 5,
    device: str = "cpu",
) -> list[Path]:
    """
    Run inference on a random sample of patches and generate all three plot types.

    Args:
        model:        Trained SR model (eval mode expected)
        landsat_dir:  Directory with Landsat .npy patches
        hyperion_dir: Directory with Hyperion .npy patches
        output_dir:   Directory to save PNG files
        wav_hyp:      Hyperion wavelengths (175,)
        wav_landsat:  Landsat wavelengths  (7,)
        num_samples:  How many patches to visualize
        device:       Torch device string

    Returns:
        List of saved PNG paths
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    landsat_files = sorted(Path(landsat_dir).glob("*.npy"))
    hyperion_files = sorted(Path(hyperion_dir).glob("*.npy"))

    if not landsat_files:
        logger.warning("No Landsat patches found in %s", landsat_dir)
        return []

    indices = np.random.choice(len(landsat_files), min(num_samples, len(landsat_files)),
                               replace=False)
    saved_paths = []

    model.eval()

    for idx in indices:
        ls_file = landsat_files[idx]
        hy_file = hyperion_files[idx] if idx < len(hyperion_files) else None

        scene_name = ls_file.stem

        patch_landsat = np.load(ls_file)          # (H, W, 7)
        patch_hyp = np.load(hy_file) if hy_file and hy_file.exists() else None

        # Reconstruct
        inp = torch.from_numpy(patch_landsat).float().permute(2, 0, 1).unsqueeze(0)
        inp = inp.to(device)

        with torch.no_grad():
            out = model(inp)

        patch_recon = out.reshape(175, 64, 64).permute(1, 2, 0).cpu().numpy()  # (H,W,175)

        row = np.random.randint(0, patch_landsat.shape[0])
        col = np.random.randint(0, patch_landsat.shape[1])

        # 1. Spectral curve
        if patch_hyp is not None:
            curve_png = spectral_curve_plot(patch_hyp, patch_landsat, patch_recon,
                                            wav_hyp, wav_landsat, row, col, scene_name)
            p = output_dir / f"{scene_name}_spectral_curve.png"
            p.write_bytes(curve_png)
            saved_paths.append(p)

            # 2. Band distribution
            dist_png = band_distribution_plot(patch_hyp, patch_recon, scene_name=scene_name)
            p = output_dir / f"{scene_name}_band_distribution.png"
            p.write_bytes(dist_png)
            saved_paths.append(p)

            # 3. False-color composite
            fc_png = false_color_composite_plot(patch_hyp, patch_recon,
                                                row=row, col=col, scene_name=scene_name)
            p = output_dir / f"{scene_name}_false_color.png"
            p.write_bytes(fc_png)
            saved_paths.append(p)

            logger.info("Generated plots for %s", scene_name)

    return saved_paths
