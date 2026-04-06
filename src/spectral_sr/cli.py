"""Command-line interface for spectral super-resolution."""

import argparse
import sys
from typing import List

from .config import ModelConfig, DataConfig, TrainConfig, InferenceConfig
from .train import Trainer
from .inference import reconstruct_from_config


def parse_encoder_channels(s: str) -> List[int]:
    """Parse comma-separated encoder channel sizes."""
    try:
        channels = [int(x.strip()) for x in s.split(',')]
        if len(channels) != 3:
            raise ValueError("Must provide exactly 3 channel sizes")
        return channels
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid encoder channels format. Expected 3 comma-separated integers, got: {s}"
        )


def parse_block_coeff(s: str) -> List[float]:
    """Parse comma-separated block coefficients."""
    try:
        coeff = [float(x.strip()) for x in s.split(',')]
        if len(coeff) != 3:
            raise ValueError("Must provide exactly 3 coefficients")
        return coeff
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid block coefficients format. Expected 3 comma-separated floats, got: {s}"
        )


def train_command(args):
    """Execute training command."""
    # Create model config
    model_config = ModelConfig(
        model_type=args.model,
        in_channels=175 if args.mode == 'autoencoder' else 7,
        out_channels=175,
        encoder_channels=args.encoder_channels
    )
    
    # Create data config
    data_config = DataConfig(
        hyperion_train_dir=args.hyperion_train_dir,
        hyperion_val_dir=args.hyperion_val_dir,
        landsat_train_dir=args.landsat_train_dir,
        landsat_val_dir=args.landsat_val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create train config
    train_config = TrainConfig(
        mode=args.mode,
        loss_type=args.loss,
        ae_checkpoint=args.ae_checkpoint,
        epochs=args.epochs,
        lr=args.lr,
        content_loss_coeff=args.content_loss_coeff,
        style_loss_coeff=args.style_loss_coeff,
        block_coeff=args.block_coeff,
        output_dir=args.output_dir,
        checkpoint_every=args.checkpoint_every,
        device=args.device,
        seed=args.seed
    )
    
    # Create trainer and run training
    trainer = Trainer(model_config, data_config, train_config)
    trainer.train()


def reconstruct_command(args):
    """Execute reconstruction command."""
    # Create model config
    model_config = ModelConfig(
        model_type=args.model,
        in_channels=7,
        out_channels=175,
        encoder_channels=args.encoder_channels
    )
    
    # Create inference config
    inference_config = InferenceConfig(
        checkpoint=args.checkpoint,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ground_truth_dir=args.ground_truth_dir,
        device=args.device
    )
    
    # Run reconstruction
    reconstruct_from_config(model_config, inference_config)


def main(argv=None):
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Spectral Super-Resolution for satellite imagery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train autoencoder
  spectral-sr train --mode autoencoder --model hourglass \\
    --hyperion-train-dir data/hyperion_train \\
    --hyperion-val-dir data/hyperion_val \\
    --encoder-channels 150,100,75 \\
    --epochs 5 --output-dir outputs/ae

  # Train SR with MSE loss
  spectral-sr train --mode sr --model hourglass --loss mse \\
    --hyperion-train-dir data/hyperion_train \\
    --hyperion-val-dir data/hyperion_val \\
    --landsat-train-dir data/landsat_train \\
    --landsat-val-dir data/landsat_val \\
    --encoder-channels 150,100,75 \\
    --epochs 5 --output-dir outputs/sr_mse

  # Train SR with perceptual loss
  spectral-sr train --mode sr --model hourglass --loss perceptual \\
    --ae-checkpoint outputs/ae/checkpoint_epoch_final.pt \\
    --hyperion-train-dir data/hyperion_train \\
    --hyperion-val-dir data/hyperion_val \\
    --landsat-train-dir data/landsat_train \\
    --landsat-val-dir data/landsat_val \\
    --encoder-channels 150,100,75 \\
    --epochs 5 --output-dir outputs/sr_perceptual

  # Reconstruct images
  spectral-sr reconstruct --model hourglass \\
    --checkpoint outputs/sr_mse/checkpoint_epoch_final.pt \\
    --input-dir data/landsat_test \\
    --output-dir outputs/reconstructed \\
    --encoder-channels 150,100,75
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True
    
    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train a model')
    
    train_parser.add_argument('--mode', type=str, required=True,
                             choices=['autoencoder', 'sr'],
                             help='Training mode: autoencoder (Hyperion->Hyperion) or sr (Landsat->Hyperion)')
    train_parser.add_argument('--model', type=str, required=True,
                             choices=['hourglass', 'koundinya'],
                             help='Model architecture')
    train_parser.add_argument('--loss', type=str, default='mse',
                             choices=['mse', 'perceptual'],
                             help='Loss function (default: mse)')
    train_parser.add_argument('--ae-checkpoint', type=str, default=None,
                             help='Path to pretrained autoencoder (required for perceptual loss)')
    
    train_parser.add_argument('--hyperion-train-dir', type=str, required=True,
                             help='Path to Hyperion training patches (.npy)')
    train_parser.add_argument('--hyperion-val-dir', type=str, required=True,
                             help='Path to Hyperion validation patches (.npy)')
    train_parser.add_argument('--landsat-train-dir', type=str, default=None,
                             help='Path to Landsat training patches (.npy, required for SR mode)')
    train_parser.add_argument('--landsat-val-dir', type=str, default=None,
                             help='Path to Landsat validation patches (.npy, required for SR mode)')
    
    train_parser.add_argument('--epochs', type=int, default=5,
                             help='Number of training epochs (default: 5)')
    train_parser.add_argument('--lr', type=float, default=1e-3,
                             help='Learning rate (default: 0.001)')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Batch size (default: 32)')
    train_parser.add_argument('--num-workers', type=int, default=4,
                             help='Number of data loader workers (default: 4)')
    
    train_parser.add_argument('--encoder-channels', type=parse_encoder_channels,
                             default=[150, 100, 75],
                             help='Encoder layer sizes as comma-separated integers (default: 150,100,75)')
    train_parser.add_argument('--content-loss-coeff', type=float, default=1.0,
                             help='Content loss coefficient (alpha, default: 1.0)')
    train_parser.add_argument('--style-loss-coeff', type=float, default=1e-3,
                             help='Style/perceptual loss coefficient (beta, default: 0.001)')
    train_parser.add_argument('--block-coeff', type=parse_block_coeff,
                             default=[1.0, 1.0, 1.0],
                             help='Block coefficients as comma-separated floats (default: 1.0,1.0,1.0)')
    
    train_parser.add_argument('--output-dir', type=str, default='./outputs',
                             help='Output directory for checkpoints and logs (default: ./outputs)')
    train_parser.add_argument('--checkpoint-every', type=int, default=1,
                             help='Save checkpoint every N epochs (default: 1)')
    train_parser.add_argument('--device', type=str, default=None,
                             choices=['cuda', 'cpu'],
                             help='Device to use (default: auto-detect)')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed for reproducibility (default: 42)')
    
    train_parser.set_defaults(func=train_command)
    
    # Reconstruct subcommand
    recon_parser = subparsers.add_parser('reconstruct', help='Reconstruct hyperspectral images')
    
    recon_parser.add_argument('--model', type=str, required=True,
                             choices=['hourglass', 'koundinya'],
                             help='Model architecture')
    recon_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Path to trained model checkpoint (.pt)')
    recon_parser.add_argument('--input-dir', type=str, required=True,
                             help='Directory with multispectral input patches (.npy)')
    recon_parser.add_argument('--output-dir', type=str, required=True,
                             help='Directory to save reconstructed patches')
    recon_parser.add_argument('--ground-truth-dir', type=str, default=None,
                             help='Optional directory with ground truth for metrics')
    
    recon_parser.add_argument('--encoder-channels', type=parse_encoder_channels,
                             default=[150, 100, 75],
                             help='Encoder layer sizes (must match trained model, default: 150,100,75)')
    recon_parser.add_argument('--device', type=str, default=None,
                             choices=['cuda', 'cpu'],
                             help='Device to use (default: auto-detect)')
    
    recon_parser.set_defaults(func=reconstruct_command)
    
    # Parse arguments and execute command
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
