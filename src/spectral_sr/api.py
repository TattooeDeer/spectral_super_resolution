"""FastAPI REST API for spectral super-resolution training and inference.

Exposes two long-running operations as async background jobs:
  POST /train        – train autoencoder or SR model
  POST /reconstruct  – reconstruct hyperspectral images from multispectral patches

Results (checkpoints, logs, plots) are uploaded automatically to Cloudflare R2.

Usage:
    uvicorn spectral_sr.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .config import DataConfig, InferenceConfig, ModelConfig, TrainConfig
from .inference import SpectralReconstructor
from .storage import list_experiments, upload_directory, upload_json
from .train import Trainer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Spectral Super-Resolution API",
    description=(
        "REST API for training and running spectral reconstruction models "
        "(Landsat-8 OLI → EO-1 Hyperion) with optional spectral-aware perceptual loss."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Optional API-key auth (set API_KEY env var to enable)
# ---------------------------------------------------------------------------

_API_KEY = os.getenv("API_KEY", "")
_security = HTTPBearer(auto_error=False)


def _check_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Security(_security)):
    if not _API_KEY:
        return  # auth disabled
    if credentials is None or credentials.credentials != _API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# In-memory job store (adequate for single-pod deployments)
# ---------------------------------------------------------------------------

_jobs: Dict[str, Dict[str, Any]] = {}


def _new_job(kind: str, request_body: dict) -> dict:
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "kind": kind,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "request": request_body,
        "r2_prefix": None,
        "r2_keys": [],
        "error": None,
        "metrics": None,
    }
    _jobs[job_id] = job
    return job


def _update_job(job_id: str, **kwargs):
    _jobs[job_id].update({"updated_at": datetime.now(timezone.utc).isoformat(), **kwargs})


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    mode: Literal["autoencoder", "sr"] = Field(...,
        description="autoencoder: Hyperion→Hyperion | sr: Landsat→Hyperion")
    model: Literal["hourglass", "koundinya"] = Field("hourglass")
    loss: Literal["mse", "perceptual"] = Field("mse")
    ae_checkpoint: Optional[str] = Field(None,
        description="Server-side path to pretrained AE (required for perceptual loss)")

    hyperion_train_dir: str = Field(..., description="Server-side path to Hyperion train patches")
    hyperion_val_dir: str = Field(..., description="Server-side path to Hyperion val patches")
    landsat_train_dir: Optional[str] = Field(None)
    landsat_val_dir: Optional[str] = Field(None)

    encoder_channels: List[int] = Field([150, 100, 75],
        description="Encoder channel sizes [upper, middle, lower]")
    epochs: int = Field(5, ge=1)
    lr: float = Field(1e-3, gt=0)
    batch_size: int = Field(32, ge=1)
    num_workers: int = Field(4, ge=0)
    content_loss_coeff: float = Field(1.0, gt=0)
    style_loss_coeff: float = Field(1e-3, gt=0)
    block_coeff: List[float] = Field([1.0, 1.0, 1.0])
    seed: int = Field(42)

    experiment_name: str = Field("experiment",
        description="Label used as the R2 folder prefix")


class ReconstructRequest(BaseModel):
    model: Literal["hourglass", "koundinya"] = Field("hourglass")
    checkpoint: str = Field(..., description="Server-side path to trained model .pt file")
    input_dir: str = Field(..., description="Server-side path to Landsat patches")
    encoder_channels: List[int] = Field([150, 100, 75])
    ground_truth_dir: Optional[str] = Field(None,
        description="Server-side path to Hyperion patches (for metrics)")

    # Optional wavelength metadata for spectral plots
    hyperion_properties_path: Optional[str] = Field(None)
    landsat_properties_path: Optional[str] = Field(None)
    num_plot_samples: int = Field(5, ge=0, le=50,
        description="Number of patches to visualize and upload")

    experiment_name: str = Field("reconstruction")


class JobStatusResponse(BaseModel):
    job_id: str
    kind: str
    status: str
    created_at: str
    updated_at: str
    r2_prefix: Optional[str]
    r2_keys: List[str]
    error: Optional[str]
    metrics: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Background worker functions
# ---------------------------------------------------------------------------

def _run_training(job_id: str, req: TrainRequest):
    """Execute training in background and upload results to R2."""
    try:
        _update_job(job_id, status="running")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        r2_prefix = f"{req.experiment_name}/{timestamp}"
        output_dir = Path(f"/tmp/spectral_sr_jobs/{job_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Upload request config immediately
        _update_job(job_id, r2_prefix=r2_prefix)
        upload_json(req.model_dump(), f"{r2_prefix}/config.json")

        model_config = ModelConfig(
            model_type=req.model,
            in_channels=175 if req.mode == "autoencoder" else 7,
            out_channels=175,
            encoder_channels=req.encoder_channels,
        )
        data_config = DataConfig(
            hyperion_train_dir=req.hyperion_train_dir,
            hyperion_val_dir=req.hyperion_val_dir,
            landsat_train_dir=req.landsat_train_dir,
            landsat_val_dir=req.landsat_val_dir,
            batch_size=req.batch_size,
            num_workers=req.num_workers,
        )
        train_config = TrainConfig(
            mode=req.mode,
            loss_type=req.loss,
            ae_checkpoint=req.ae_checkpoint,
            epochs=req.epochs,
            lr=req.lr,
            content_loss_coeff=req.content_loss_coeff,
            style_loss_coeff=req.style_loss_coeff,
            block_coeff=req.block_coeff,
            output_dir=str(output_dir),
            checkpoint_every=1,
            device=None,  # auto-detect
            seed=req.seed,
        )

        trainer = Trainer(model_config, data_config, train_config)
        trainer.train()

        # Upload all outputs to R2
        keys = upload_directory(output_dir, r2_prefix)

        _update_job(job_id, status="done", r2_keys=keys,
                    metrics=trainer.history)
        logger.info("Job %s completed. Uploaded %d files to R2.", job_id, len(keys))

    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        _update_job(job_id, status="failed", error=traceback.format_exc())


def _run_reconstruction(job_id: str, req: ReconstructRequest):
    """Execute reconstruction in background and upload results + plots to R2."""
    try:
        _update_job(job_id, status="running")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        r2_prefix = f"{req.experiment_name}/{timestamp}"
        output_dir = Path(f"/tmp/spectral_sr_jobs/{job_id}")
        patches_dir = output_dir / "patches"
        plots_dir = output_dir / "plots"
        patches_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        _update_job(job_id, r2_prefix=r2_prefix)
        upload_json(req.model_dump(), f"{r2_prefix}/config.json")

        model_config = ModelConfig(
            model_type=req.model,
            in_channels=7,
            out_channels=175,
            encoder_channels=req.encoder_channels,
        )
        inference_config = InferenceConfig(
            checkpoint=req.checkpoint,
            input_dir=req.input_dir,
            output_dir=str(patches_dir),
            ground_truth_dir=req.ground_truth_dir,
            device=None,
        )

        reconstructor = SpectralReconstructor(model_config, inference_config)
        metrics = reconstructor.reconstruct()

        # Generate spectral plots if wavelength metadata is provided
        if req.num_plot_samples > 0 and req.hyperion_properties_path \
                and req.landsat_properties_path and req.ground_truth_dir:
            try:
                from .visualization import (
                    generate_reconstruction_plots,
                    load_hyperion_wavelengths,
                    load_landsat_wavelengths,
                )

                device = "cuda" if torch.cuda.is_available() else "cpu"
                wav_hyp = load_hyperion_wavelengths(req.hyperion_properties_path)
                wav_ls = load_landsat_wavelengths(req.landsat_properties_path)

                generate_reconstruction_plots(
                    model=reconstructor.model,
                    landsat_dir=req.input_dir,
                    hyperion_dir=req.ground_truth_dir,
                    output_dir=plots_dir,
                    wav_hyp=wav_hyp,
                    wav_landsat=wav_ls,
                    num_samples=req.num_plot_samples,
                    device=device,
                )
            except Exception:
                logger.warning("Plot generation failed (non-fatal):\n%s", traceback.format_exc())

        # Upload patches (npy) and plots (png) to R2
        keys = upload_directory(patches_dir, f"{r2_prefix}/patches",
                                extensions=(".npy",))
        keys += upload_directory(plots_dir, f"{r2_prefix}/plots",
                                 extensions=(".png",))
        if metrics:
            upload_json(metrics, f"{r2_prefix}/metrics.json")
            keys.append(f"{r2_prefix}/metrics.json")

        _update_job(job_id, status="done", r2_keys=keys, metrics=metrics)
        logger.info("Job %s completed. Uploaded %d files to R2.", job_id, len(keys))

    except Exception:
        logger.exception("Job %s failed", job_id)
        _update_job(job_id, status="failed", error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health():
    """Return service status and device information."""
    cuda_available = torch.cuda.is_available()
    device_info = {}
    if cuda_available:
        device_info = {
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1e6, 2),
            "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1e6, 2),
        }
    return {
        "status": "ok",
        "cuda_available": cuda_available,
        "device": "cuda" if cuda_available else "cpu",
        **device_info,
    }


@app.post("/train", response_model=JobStatusResponse, status_code=202, tags=["training"],
          dependencies=[Depends(_check_api_key)])
def submit_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Submit a training job (autoencoder or spectral reconstruction).

    Returns immediately with a `job_id`. Poll `GET /jobs/{job_id}` for progress.
    When complete, model checkpoints and logs are automatically uploaded to R2.
    """
    job = _new_job("train", req.model_dump())
    background_tasks.add_task(_run_training, job["job_id"], req)
    logger.info("Queued training job %s (mode=%s, loss=%s)", job["job_id"], req.mode, req.loss)
    return JobStatusResponse(**job)


@app.post("/reconstruct", response_model=JobStatusResponse, status_code=202, tags=["inference"],
          dependencies=[Depends(_check_api_key)])
def submit_reconstruction(req: ReconstructRequest, background_tasks: BackgroundTasks):
    """
    Submit a reconstruction job.

    Runs the trained SR model over all `.npy` patches in `input_dir`, saves
    reconstructed patches and spectral plots, then uploads everything to R2.
    """
    job = _new_job("reconstruct", req.model_dump())
    background_tasks.add_task(_run_reconstruction, job["job_id"], req)
    logger.info("Queued reconstruction job %s", job["job_id"])
    return JobStatusResponse(**job)


@app.get("/jobs", response_model=List[JobStatusResponse], tags=["jobs"],
         dependencies=[Depends(_check_api_key)])
def list_jobs(kind: Optional[str] = None, status_filter: Optional[str] = None):
    """List all jobs, optionally filtered by `kind` (train/reconstruct) or `status`."""
    jobs = list(_jobs.values())
    if kind:
        jobs = [j for j in jobs if j["kind"] == kind]
    if status_filter:
        jobs = [j for j in jobs if j["status"] == status_filter]
    return [JobStatusResponse(**j) for j in sorted(jobs, key=lambda j: j["created_at"])]


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"],
         dependencies=[Depends(_check_api_key)])
def get_job(job_id: str):
    """Get the status and results for a specific job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JobStatusResponse(**_jobs[job_id])


@app.get("/experiments", tags=["storage"], dependencies=[Depends(_check_api_key)])
def get_experiments():
    """List all top-level experiment folders saved in R2."""
    try:
        experiments = list_experiments()
        return {"experiments": experiments, "count": len(experiments)}
    except Exception as exc:
        raise HTTPException(status_code=503,
                            detail=f"Could not reach R2 storage: {exc}") from exc


# ---------------------------------------------------------------------------
# Entrypoint helper (used by the `spectral-sr-api` console script)
# ---------------------------------------------------------------------------

def start_server():
    """Start the API server with uvicorn (called via CLI)."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info")

    uvicorn.run(
        "spectral_sr.api:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
    )


if __name__ == "__main__":
    start_server()
