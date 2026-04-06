# Spectral Super-Resolution

Deep learning-based spectral super-resolution for satellite imagery. This repository implements the methods described in "Spectral Recovery Via Spectral-Aware Perceptual Loss" for reconstructing hyperspectral images from multispectral inputs.

## Overview

This project enables:
- **Spectral Reconstruction**: Transform Landsat-8/OLI 7-band multispectral images into EO-1/Hyperion 175-band hyperspectral equivalents
- **Autoencoder Pre-training**: Learn spectral characteristics of Hyperion imagery
- **Perceptual Loss Training**: Novel loss function using Gram matrices from pretrained autoencoders to capture spectral fidelity

## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 2.0.0
- UV package manager (recommended)

### Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd spectral_super_resolution

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in development mode
uv pip install -e .
```

### Using pip

```bash
pip install -e .
```

## Usage

The package provides a command-line interface `spectral-sr` with two main commands: `train` and `reconstruct`.

### Training an Autoencoder

First, train an autoencoder on Hyperion images to learn spectral characteristics:

```bash
spectral-sr train \
  --mode autoencoder \
  --model hourglass \
  --hyperion-train-dir /path/to/hyperion_train_npy \
  --hyperion-val-dir /path/to/hyperion_val_npy \
  --encoder-channels 150,100,75 \
  --epochs 5 \
  --lr 0.001 \
  --batch-size 32 \
  --output-dir outputs/autoencoder
```

### Training Spectral Reconstruction with MSE

Train a model to map Landsat to Hyperion using standard MSE loss:

```bash
spectral-sr train \
  --mode sr \
  --model hourglass \
  --loss mse \
  --hyperion-train-dir /path/to/hyperion_train_npy \
  --hyperion-val-dir /path/to/hyperion_val_npy \
  --landsat-train-dir /path/to/landsat_train_npy \
  --landsat-val-dir /path/to/landsat_val_npy \
  --encoder-channels 150,100,75 \
  --epochs 5 \
  --output-dir outputs/sr_mse
```

### Training with Perceptual Loss

Train with the spectral-aware perceptual loss using a pretrained autoencoder:

```bash
spectral-sr train \
  --mode sr \
  --model hourglass \
  --loss perceptual \
  --ae-checkpoint outputs/autoencoder/checkpoint_epoch_final.pt \
  --hyperion-train-dir /path/to/hyperion_train_npy \
  --hyperion-val-dir /path/to/hyperion_val_npy \
  --landsat-train-dir /path/to/landsat_train_npy \
  --landsat-val-dir /path/to/landsat_val_npy \
  --encoder-channels 150,100,75 \
  --epochs 5 \
  --content-loss-coeff 1.0 \
  --style-loss-coeff 0.001 \
  --output-dir outputs/sr_perceptual
```

### Reconstruction (Inference)

Reconstruct hyperspectral images from multispectral inputs:

```bash
spectral-sr reconstruct \
  --model hourglass \
  --checkpoint outputs/sr_mse/checkpoint_epoch_final.pt \
  --input-dir /path/to/landsat_test_npy \
  --output-dir outputs/reconstructed \
  --encoder-channels 150,100,75 \
  --ground-truth-dir /path/to/hyperion_test_npy  # Optional, for metrics
```

## Arguments Reference

### Common Arguments

- `--model`: Model architecture (`hourglass` or `koundinya`)
- `--encoder-channels`: Comma-separated encoder layer sizes (e.g., `150,100,75` or `90,45,10`)
- `--device`: Device to use (`cuda` or `cpu`, auto-detected if not specified)

### Training Arguments

- `--mode`: Training mode (`autoencoder` or `sr`)
- `--loss`: Loss function (`mse` or `perceptual`)
- `--ae-checkpoint`: Path to pretrained autoencoder (required for perceptual loss)
- `--epochs`: Number of training epochs (default: 5)
- `--lr`: Learning rate (default: 0.001)
- `--batch-size`: Batch size (default: 32)
- `--content-loss-coeff`: Content loss weight α (default: 1.0)
- `--style-loss-coeff`: Style/perceptual loss weight β (default: 0.001)
- `--output-dir`: Directory for checkpoints and logs (default: ./outputs)
- `--seed`: Random seed for reproducibility (default: 42)

### Reconstruction Arguments

- `--checkpoint`: Path to trained model checkpoint
- `--input-dir`: Directory with multispectral .npy patches
- `--output-dir`: Directory to save reconstructed patches
- `--ground-truth-dir`: Optional directory with ground truth for computing metrics

## Data Format

The package expects data as `.npy` files (NumPy arrays) with shape `(height, width, channels)`:
- **Landsat patches**: (64, 64, 7)
- **Hyperion patches**: (64, 64, 175)

Pixel values should represent reflectance (typically scaled by 10,000 in the original data).

## Architecture

The repository uses a modular structure:

```
spectral_super_resolution/
├── src/spectral_sr/
│   ├── models.py          # Hourglass and Koundinya2D_CNN architectures
│   ├── losses.py          # GramLoss with corrected batched Gram matrix
│   ├── datasets.py        # PyTorch datasets for loading .npy patches
│   ├── metrics.py         # SSIM, SAM (Spectral Angle Mapper), MSE, RMSE
│   ├── train.py           # Unified Trainer class
│   ├── inference.py       # Reconstruction/inference
│   ├── config.py          # Configuration dataclasses
│   └── cli.py             # Command-line interface
├── legacy/                 # Original notebook-based implementation (backup)
└── pyproject.toml         # Package configuration
```

## Key Improvements in This Version

1. **Fixed Gram Matrix Bug**: Corrected batched computation in `gram_matrix()` to compute per-sample Gram matrices rather than mixing inter-sample correlations
2. **Modular Design**: Separated concerns into models, losses, datasets, training, and inference
3. **CLI Interface**: Easy-to-use command-line tool for training and reconstruction
4. **Configuration Management**: Type-safe configs with validation
5. **Cross-platform**: Path handling works on Windows, macOS, and Linux

## Citation

If you use this code, please cite:

```
[Add citation for your paper once published]
```

## Original Notebook

The original Jupyter notebook implementation is preserved in the `legacy/` directory for reference.

## Author

Ignacio Loayza Campos

## License

[Add license information]
