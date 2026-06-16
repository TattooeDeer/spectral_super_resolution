"""Command-line interface for spectral super-resolution.

R2 integration
--------------
All directory / file path arguments accept either:
  - A local filesystem path          e.g.  /data/hyperion_train_npy
  - An R2 object reference           e.g.  r2://hyperion_train_npy

When an ``r2://`` path is given the object (or directory prefix) is
downloaded to a temporary local directory before the command runs.

R2 credentials are resolved in this order:
  1. CLI flags  --r2-access-key-id / --r2-secret-access-key / --r2-endpoint-url
  2. Environment variables  R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_ENDPOINT_URL
  3. .env file in the current working directory

Upload
------
Pass ``--r2-upload`` (with ``--r2-experiment-name``) to automatically
upload all outputs to R2 after the command completes.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import List, Optional

from .storage import R2Config, resolve_directory, resolve_path, upload_directory


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_encoder_channels(s: str) -> List[int]:
    try:
        ch = [int(x.strip()) for x in s.split(",")]
        if len(ch) != 3:
            raise ValueError()
        return ch
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Expected 3 comma-separated integers (e.g. 150,100,75), got: {s!r}"
        )


def parse_block_coeff(s: str) -> List[float]:
    try:
        coeff = [float(x.strip()) for x in s.split(",")]
        if len(coeff) != 3:
            raise ValueError()
        return coeff
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Expected 3 comma-separated floats (e.g. 1.0,1.0,1.0), got: {s!r}"
        )


# ---------------------------------------------------------------------------
# R2Config from parsed CLI args
# ---------------------------------------------------------------------------

def _r2_cfg(args) -> R2Config:
    """Build an R2Config by overlaying CLI flags on top of env vars / .env."""
    cfg = R2Config.from_env()
    if getattr(args, "r2_access_key_id", None):
        cfg.access_key_id = args.r2_access_key_id
    if getattr(args, "r2_secret_access_key", None):
        cfg.secret_access_key = args.r2_secret_access_key
    if getattr(args, "r2_endpoint_url", None):
        cfg.endpoint_url = args.r2_endpoint_url
    if getattr(args, "r2_bucket", None):
        cfg.bucket = args.r2_bucket
    return cfg


def _resolve_dir(path: Optional[str], cfg: R2Config, tmp: Path,
                 label: str) -> Optional[str]:
    if path is None:
        return None
    if path.startswith("r2://"):
        key = path[len("r2://"):]
        local = str(tmp / label)
        print(f"[R2] Downloading r2://{key} -> {local}")
        return resolve_directory(key, cfg=cfg, local_dir=local)
    return path


def _resolve_file(path: Optional[str], cfg: R2Config, tmp: Path) -> Optional[str]:
    if path is None:
        return None
    if path.startswith("r2://"):
        print(f"[R2] Downloading {path}")
        return resolve_path(path, cfg=cfg, tmp_dir=str(tmp))
    return path


# ---------------------------------------------------------------------------
# R2 argument group (shared by both subcommands)
# ---------------------------------------------------------------------------

def _add_r2_args(parser: argparse.ArgumentParser):
    r2 = parser.add_argument_group(
        "R2 credentials",
        "Override the R2 credentials from .env / environment variables."
    )
    r2.add_argument("--r2-access-key-id", metavar="KEY",
                    help="R2 Access Key ID (overrides R2_ACCESS_KEY_ID env var)")
    r2.add_argument("--r2-secret-access-key", metavar="SECRET",
                    help="R2 Secret Access Key (overrides R2_SECRET_ACCESS_KEY env var)")
    r2.add_argument("--r2-endpoint-url", metavar="URL",
                    help="R2 endpoint URL (overrides R2_ENDPOINT_URL env var)")
    r2.add_argument("--r2-bucket", metavar="BUCKET",
                    help="R2 bucket name (overrides R2_BUCKET_NAME env var)")

    up = parser.add_argument_group("R2 upload", "Upload outputs to R2 after completion.")
    up.add_argument("--r2-upload", action="store_true",
                    help="Upload all outputs to R2 after the command completes")
    up.add_argument("--r2-experiment-name", metavar="NAME", default="experiment",
                    help="R2 folder prefix for the uploaded outputs (default: experiment)")


# ---------------------------------------------------------------------------
# train command
# ---------------------------------------------------------------------------

def train_command(args):
    from .config import DataConfig, ModelConfig, TrainConfig
    from .train import Trainer

    cfg = _r2_cfg(args)

    with tempfile.TemporaryDirectory(prefix="spectral_sr_") as tmp_str:
        tmp = Path(tmp_str)

        ae_local = _resolve_file(args.ae_checkpoint, cfg, tmp)
        hyp_train = _resolve_dir(args.hyperion_train_dir, cfg, tmp, "hyperion_train")
        hyp_val = _resolve_dir(args.hyperion_val_dir, cfg, tmp, "hyperion_val")
        ls_train = _resolve_dir(args.landsat_train_dir, cfg, tmp, "landsat_train")
        ls_val = _resolve_dir(args.landsat_val_dir, cfg, tmp, "landsat_val")

        model_config = ModelConfig(
            model_type=args.model,
            in_channels=175 if args.mode == "autoencoder" else 7,
            out_channels=175,
            encoder_channels=args.encoder_channels,
        )
        data_config = DataConfig(
            hyperion_train_dir=hyp_train,
            hyperion_val_dir=hyp_val,
            landsat_train_dir=ls_train,
            landsat_val_dir=ls_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        train_config = TrainConfig(
            mode=args.mode,
            loss_type=args.loss,
            ae_checkpoint=ae_local,
            epochs=args.epochs,
            lr=args.lr,
            content_loss_coeff=args.content_loss_coeff,
            style_loss_coeff=args.style_loss_coeff,
            block_coeff=args.block_coeff,
            output_dir=args.output_dir,
            checkpoint_every=args.checkpoint_every,
            device=args.device,
            seed=args.seed,
        )

        trainer = Trainer(model_config, data_config, train_config)
        trainer.train()

    if args.r2_upload:
        print(f"\n[R2] Uploading outputs to r2://{args.r2_experiment_name}/")
        cfg.validate()
        keys = upload_directory(Path(args.output_dir), args.r2_experiment_name, cfg=cfg)
        print(f"[R2] Uploaded {len(keys)} files.")


# ---------------------------------------------------------------------------
# reconstruct command
# ---------------------------------------------------------------------------

def reconstruct_command(args):
    from .config import InferenceConfig, ModelConfig
    from .inference import reconstruct_from_config

    cfg = _r2_cfg(args)

    with tempfile.TemporaryDirectory(prefix="spectral_sr_") as tmp_str:
        tmp = Path(tmp_str)

        ckpt_local = _resolve_file(args.checkpoint, cfg, tmp)
        input_local = _resolve_dir(args.input_dir, cfg, tmp, "input")
        gt_local = _resolve_dir(args.ground_truth_dir, cfg, tmp, "ground_truth")

        model_config = ModelConfig(
            model_type=args.model,
            in_channels=7,
            out_channels=175,
            encoder_channels=args.encoder_channels,
        )
        inference_config = InferenceConfig(
            checkpoint=ckpt_local,
            input_dir=input_local,
            output_dir=args.output_dir,
            ground_truth_dir=gt_local,
            device=args.device,
        )

        reconstruct_from_config(model_config, inference_config)

    if args.r2_upload:
        print(f"\n[R2] Uploading outputs to r2://{args.r2_experiment_name}/")
        cfg.validate()
        keys = upload_directory(Path(args.output_dir), args.r2_experiment_name, cfg=cfg)
        print(f"[R2] Uploaded {len(keys)} files.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spectral Super-Resolution – Landsat-8 OLI → EO-1 Hyperion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (local paths):
  spectral-sr train --mode autoencoder --model hourglass \\
    --hyperion-train-dir /data/hyperion_train \\
    --hyperion-val-dir   /data/hyperion_val  \\
    --encoder-channels 150,100,75 --epochs 5 --output-dir outputs/ae

  spectral-sr train --mode sr --loss perceptual \\
    --ae-checkpoint      r2://Models/AEHG_150_100_75.pt \\
    --hyperion-train-dir r2://hyperion_train_npy \\
    --hyperion-val-dir   r2://hyperion_val_npy   \\
    --landsat-train-dir  r2://landsat_train_npy  \\
    --landsat-val-dir    r2://landsat_val_npy    \\
    --encoder-channels 150,100,75 --epochs 5          \\
    --output-dir outputs/sr --r2-upload               \\
    --r2-experiment-name my_experiment

  spectral-sr reconstruct --model hourglass \\
    --checkpoint  r2://Models/hg_SR_150_100_75.pt \\
    --input-dir   r2://landsat_test_npy      \\
    --output-dir  outputs/reconstructed           \\
    --r2-upload --r2-experiment-name my_reconstruction
        """,
    )

    sub = parser.add_subparsers(dest="command", help="Command to run")
    sub.required = True

    # ── train ──────────────────────────────────────────────────────────────

    train_p = sub.add_parser("train", help="Train a model",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    train_p.add_argument("--mode", required=True, choices=["autoencoder", "sr"],
                         help="autoencoder: Hyperion→Hyperion | sr: Landsat→Hyperion")
    train_p.add_argument("--model", required=True, choices=["hourglass", "koundinya"],
                         help="Model architecture")
    train_p.add_argument("--loss", default="mse", choices=["mse", "perceptual"])
    train_p.add_argument("--ae-checkpoint", default=None,
                         help="Local path or r2://key of pretrained AE "
                              "(required for --loss perceptual)")

    data_grp = train_p.add_argument_group("data paths (local or r2://)")
    data_grp.add_argument("--hyperion-train-dir", required=True)
    data_grp.add_argument("--hyperion-val-dir", required=True)
    data_grp.add_argument("--landsat-train-dir", default=None)
    data_grp.add_argument("--landsat-val-dir", default=None)

    hp = train_p.add_argument_group("hyperparameters")
    hp.add_argument("--epochs", type=int, default=5)
    hp.add_argument("--lr", type=float, default=1e-3)
    hp.add_argument("--batch-size", type=int, default=32)
    hp.add_argument("--num-workers", type=int, default=4)
    hp.add_argument("--encoder-channels", type=parse_encoder_channels, default=[150, 100, 75],
                    metavar="A,B,C")
    hp.add_argument("--content-loss-coeff", type=float, default=1.0, metavar="α")
    hp.add_argument("--style-loss-coeff", type=float, default=1e-3, metavar="β")
    hp.add_argument("--block-coeff", type=parse_block_coeff, default=[1.0, 1.0, 1.0],
                    metavar="A,B,C")
    hp.add_argument("--seed", type=int, default=42)

    out_grp = train_p.add_argument_group("output")
    out_grp.add_argument("--output-dir", default="./outputs")
    out_grp.add_argument("--checkpoint-every", type=int, default=1, metavar="N")
    out_grp.add_argument("--device", choices=["cuda", "cpu"], default=None)

    _add_r2_args(train_p)
    train_p.set_defaults(func=train_command)

    # ── reconstruct ────────────────────────────────────────────────────────

    rec_p = sub.add_parser("reconstruct", help="Reconstruct hyperspectral images",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    rec_p.add_argument("--model", required=True, choices=["hourglass", "koundinya"])
    rec_p.add_argument("--checkpoint", required=True,
                       help="Local path or r2://key of trained model .pt file")

    rec_data = rec_p.add_argument_group("data paths (local or r2://)")
    rec_data.add_argument("--input-dir", required=True,
                          help="Multispectral .npy patch directory")
    rec_data.add_argument("--output-dir", required=True,
                          help="Local directory to save reconstructed patches")
    rec_data.add_argument("--ground-truth-dir", default=None,
                          help="Hyperspectral .npy patches for metric computation")

    rec_p.add_argument("--encoder-channels", type=parse_encoder_channels, default=[150, 100, 75],
                       metavar="A,B,C",
                       help="Must match the trained model (default: 150,100,75)")
    rec_p.add_argument("--device", choices=["cuda", "cpu"], default=None)

    _add_r2_args(rec_p)
    rec_p.set_defaults(func=reconstruct_command)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
