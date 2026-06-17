"""FastAPI REST API for spectral super-resolution training and inference.

Credentials and path resolution
---------------------------------
R2 credentials may be supplied in three ways (highest priority first):
  1. Inline in the JSON request body  (``r2`` block)
  2. Environment variables (SPECTRAL_RECONSTRUCTION_R2_ACCESS_KEY_ID / …)
  3. A .env file in the working directory (loaded automatically)

Any path field that starts with ``r2://`` is downloaded to a temporary
local directory before the job starts, so models and dataset directories
can live entirely in R2.

Usage:
    uvicorn spectral_sr.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, model_validator

from .config import DataConfig, InferenceConfig, ModelConfig, TrainConfig
from . import env
from .version import __version__
from .inference import SpectralReconstructor
from .storage import R2Config, list_experiments, resolve_directory, resolve_path
from .storage import upload_directory, upload_json
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
        "(Landsat-8 OLI → EO-1 Hyperion) with optional spectral-aware perceptual loss.\n\n"
        "**R2 paths:** Any path field accepts `r2://key` to reference an object "
        "in your Cloudflare R2 bucket. It will be downloaded automatically before use."
    ),
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Optional API-key auth – set SPECTRAL_RECONSTRUCTION_API_KEY env var to enable
# ---------------------------------------------------------------------------

_security = HTTPBearer(auto_error=False)


def _check_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Security(_security)):
    api_key = env.getenv(env.API_KEY)
    if not api_key:
        return  # auth disabled
    if credentials is None or credentials.credentials != api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

_jobs: Dict[str, Dict[str, Any]] = {}


def _new_job(kind: str, request_body: dict) -> dict:
    job_id = str(uuid.uuid4())
    job: Dict[str, Any] = {
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
# Shared R2 credentials schema (reused by both request models)
# ---------------------------------------------------------------------------

class R2Credentials(BaseModel):
    """
    Optional per-request R2 credentials.

    If omitted the server falls back to environment variables / .env file.
    Never logged or stored in job history.
    """
    access_key_id: Optional[str] = Field(None, description="R2 Access Key ID")
    secret_access_key: Optional[str] = Field(None, description="R2 Secret Access Key")
    endpoint_url: Optional[str] = Field(None,
        description="https://<account_id>.r2.cloudflarestorage.com")
    bucket: Optional[str] = Field(None,
        description="Bucket name (default: spectral-reconstruction-data-ena)")

    def to_r2config(self) -> R2Config:
        """Build an R2Config, overlaying explicit fields on top of env vars."""
        cfg = R2Config.from_env()
        if self.access_key_id:
            cfg.access_key_id = self.access_key_id
        if self.secret_access_key:
            cfg.secret_access_key = self.secret_access_key
        if self.endpoint_url:
            cfg.endpoint_url = self.endpoint_url
        if self.bucket:
            cfg.bucket = self.bucket
        return cfg


def _safe_request_dump(req: BaseModel) -> dict:
    """Serialize request body but replace secret_access_key with a placeholder."""
    d = req.model_dump()
    if "r2" in d and d["r2"] and "secret_access_key" in (d["r2"] or {}):
        if d["r2"]["secret_access_key"]:
            d["r2"]["secret_access_key"] = "***"
    return d


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    # --- R2 credentials (optional, fall back to env) ---
    r2: Optional[R2Credentials] = Field(None,
        description="R2 credentials (falls back to env vars / .env if omitted)")

    # --- Training parameters ---
    mode: Literal["autoencoder", "sr"] = Field(...,
        description="autoencoder: Hyperion→Hyperion | sr: Landsat→Hyperion")
    model: Literal["hourglass", "koundinya"] = Field("hourglass")
    loss: Literal["mse", "perceptual"] = Field("mse")

    # Accepts local path OR r2://key
    ae_checkpoint: Optional[str] = Field(None,
        description="Local path or r2://key of pretrained AE (required for perceptual loss)")

    # Accepts local paths OR r2://prefix for entire directories
    hyperion_train_dir: str = Field(...,
        description="Local path or r2://prefix for Hyperion train patches")
    hyperion_val_dir: str = Field(...,
        description="Local path or r2://prefix for Hyperion val patches")
    landsat_train_dir: Optional[str] = Field(None,
        description="Local path or r2://prefix for Landsat train patches")
    landsat_val_dir: Optional[str] = Field(None,
        description="Local path or r2://prefix for Landsat val patches")

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
        description="Label used as the R2 output folder prefix")


class ReconstructRequest(BaseModel):
    # --- R2 credentials (optional) ---
    r2: Optional[R2Credentials] = Field(None)

    # Accepts local path OR r2://key
    model: Literal["hourglass", "koundinya"] = Field("hourglass")
    checkpoint: str = Field(...,
        description="Local path or r2://key of trained model .pt file")
    input_dir: str = Field(...,
        description="Local path or r2://prefix of Landsat patches")
    encoder_channels: List[int] = Field([150, 100, 75])
    ground_truth_dir: Optional[str] = Field(None,
        description="Local path or r2://prefix of Hyperion patches (for metrics)")

    hyperion_properties_path: Optional[str] = Field(None,
        description="Local path or r2://key of Hyperion wavelength metadata .txt")
    landsat_properties_path: Optional[str] = Field(None,
        description="Local path or r2://key of Landsat wavelength metadata .txt")
    num_plot_samples: int = Field(5, ge=0, le=50)

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
# Background worker helpers
# ---------------------------------------------------------------------------

def _resolve_dir(path: Optional[str], cfg: R2Config,
                 job_tmp: Path, label: str) -> Optional[str]:
    """Resolve a local path or r2:// prefix to a usable local directory."""
    if path is None:
        return None
    if not path.startswith("r2://"):
        return path
    key_prefix = path[len("r2://"):]
    local = str(job_tmp / label)
    logger.info("Downloading R2 directory r2://%s -> %s", key_prefix, local)
    return resolve_directory(key_prefix, cfg=cfg, local_dir=local)


def _resolve_file(path: Optional[str], cfg: R2Config,
                  job_tmp: Path) -> Optional[str]:
    """Resolve a local path or r2://key to a usable local file path."""
    if path is None:
        return None
    if not path.startswith("r2://"):
        return path
    return resolve_path(path, cfg=cfg, tmp_dir=str(job_tmp))


# ---------------------------------------------------------------------------
# Background worker: training
# ---------------------------------------------------------------------------

def _run_training(job_id: str, req: TrainRequest):
    try:
        _update_job(job_id, status="running")

        cfg = req.r2.to_r2config() if req.r2 else R2Config.from_env()
        upload_cfg = cfg.for_experiments()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        r2_prefix = f"{req.experiment_name}/{timestamp}"
        job_tmp = Path(f"/tmp/spectral_sr_jobs/{job_id}")
        output_dir = job_tmp / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        _update_job(job_id, r2_prefix=r2_prefix)

        # Upload sanitised config immediately (no secrets)
        upload_json(_safe_request_dump(req), f"{r2_prefix}/config.json", cfg=upload_cfg)

        # Resolve r2:// paths to local directories/files
        ae_local = _resolve_file(req.ae_checkpoint, cfg, job_tmp)
        hyp_train = _resolve_dir(req.hyperion_train_dir, cfg, job_tmp, "hyperion_train")
        hyp_val = _resolve_dir(req.hyperion_val_dir, cfg, job_tmp, "hyperion_val")
        ls_train = _resolve_dir(req.landsat_train_dir, cfg, job_tmp, "landsat_train")
        ls_val = _resolve_dir(req.landsat_val_dir, cfg, job_tmp, "landsat_val")

        model_config = ModelConfig(
            model_type=req.model,
            in_channels=175 if req.mode == "autoencoder" else 7,
            out_channels=175,
            encoder_channels=req.encoder_channels,
        )
        data_config = DataConfig(
            hyperion_train_dir=hyp_train,
            hyperion_val_dir=hyp_val,
            landsat_train_dir=ls_train,
            landsat_val_dir=ls_val,
            batch_size=req.batch_size,
            num_workers=req.num_workers,
        )
        train_config = TrainConfig(
            mode=req.mode,
            loss_type=req.loss,
            ae_checkpoint=ae_local,
            epochs=req.epochs,
            lr=req.lr,
            content_loss_coeff=req.content_loss_coeff,
            style_loss_coeff=req.style_loss_coeff,
            block_coeff=req.block_coeff,
            output_dir=str(output_dir),
            checkpoint_every=1,
            device=None,
            seed=req.seed,
        )

        trainer = Trainer(model_config, data_config, train_config)
        trainer.train()

        keys = upload_directory(output_dir, r2_prefix, cfg=upload_cfg)
        _update_job(job_id, status="done", r2_keys=keys, metrics=trainer.history)
        logger.info("Training job %s done. Uploaded %d files.", job_id, len(keys))

    except Exception:
        logger.exception("Training job %s failed", job_id)
        _update_job(job_id, status="failed", error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Background worker: reconstruction
# ---------------------------------------------------------------------------

def _run_reconstruction(job_id: str, req: ReconstructRequest):
    try:
        _update_job(job_id, status="running")

        cfg = req.r2.to_r2config() if req.r2 else R2Config.from_env()
        upload_cfg = cfg.for_experiments()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        r2_prefix = f"{req.experiment_name}/{timestamp}"
        job_tmp = Path(f"/tmp/spectral_sr_jobs/{job_id}")
        patches_out = job_tmp / "patches"
        plots_out = job_tmp / "plots"
        patches_out.mkdir(parents=True, exist_ok=True)
        plots_out.mkdir(parents=True, exist_ok=True)

        _update_job(job_id, r2_prefix=r2_prefix)
        upload_json(_safe_request_dump(req), f"{r2_prefix}/config.json", cfg=upload_cfg)

        # Resolve r2:// paths
        ckpt_local = _resolve_file(req.checkpoint, cfg, job_tmp)
        input_local = _resolve_dir(req.input_dir, cfg, job_tmp, "input")
        gt_local = _resolve_dir(req.ground_truth_dir, cfg, job_tmp, "ground_truth")
        hyp_props = _resolve_file(req.hyperion_properties_path, cfg, job_tmp)
        ls_props = _resolve_file(req.landsat_properties_path, cfg, job_tmp)

        model_config = ModelConfig(
            model_type=req.model,
            in_channels=7,
            out_channels=175,
            encoder_channels=req.encoder_channels,
        )
        inference_config = InferenceConfig(
            checkpoint=ckpt_local,
            input_dir=input_local,
            output_dir=str(patches_out),
            ground_truth_dir=gt_local,
            device=None,
        )

        reconstructor = SpectralReconstructor(model_config, inference_config)
        metrics = reconstructor.reconstruct()

        # Spectral plots (non-fatal if it fails)
        if req.num_plot_samples > 0 and hyp_props and ls_props and gt_local:
            try:
                from .visualization import (
                    generate_reconstruction_plots,
                    load_hyperion_wavelengths,
                    load_landsat_wavelengths,
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                generate_reconstruction_plots(
                    model=reconstructor.model,
                    landsat_dir=input_local,
                    hyperion_dir=gt_local,
                    output_dir=plots_out,
                    wav_hyp=load_hyperion_wavelengths(hyp_props),
                    wav_landsat=load_landsat_wavelengths(ls_props),
                    num_samples=req.num_plot_samples,
                    device=device,
                )
            except Exception:
                logger.warning("Plot generation failed (non-fatal):\n%s", traceback.format_exc())

        keys = upload_directory(patches_out, f"{r2_prefix}/patches",
                                cfg=upload_cfg, extensions=(".npy",))
        keys += upload_directory(plots_out, f"{r2_prefix}/plots",
                                 cfg=upload_cfg, extensions=(".png",))
        if metrics:
            upload_json(metrics, f"{r2_prefix}/metrics.json", cfg=upload_cfg)
            keys.append(f"{r2_prefix}/metrics.json")

        _update_job(job_id, status="done", r2_keys=keys, metrics=metrics)
        logger.info("Reconstruction job %s done. Uploaded %d files.", job_id, len(keys))

    except Exception:
        logger.exception("Reconstruction job %s failed", job_id)
        _update_job(job_id, status="failed", error=traceback.format_exc())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health():
    """Service status and GPU device information."""
    cuda = torch.cuda.is_available()
    info: Dict[str, Any] = {"status": "ok", "version": __version__,
                             "cuda_available": cuda,
                             "device": "cuda" if cuda else "cpu"}
    if cuda:
        info.update({
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1e6, 2),
            "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1e6, 2),
        })
    return info


@app.post("/train", response_model=JobStatusResponse, status_code=202, tags=["training"],
          dependencies=[Depends(_check_api_key)])
def submit_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Submit a training job.

    - `mode=autoencoder` trains the Hyperion→Hyperion denoising AE.
    - `mode=sr` trains the Landsat→Hyperion SR model.
    - `loss=perceptual` requires `ae_checkpoint` pointing to a trained AE.

    All directory/file paths accept `r2://key` references.
    Returns a `job_id` immediately; poll `GET /jobs/{job_id}` for status.
    """
    job = _new_job("train", _safe_request_dump(req))
    background_tasks.add_task(_run_training, job["job_id"], req)
    logger.info("Queued training job %s (mode=%s loss=%s)", job["job_id"], req.mode, req.loss)
    return JobStatusResponse(**job)


@app.post("/reconstruct", response_model=JobStatusResponse, status_code=202, tags=["inference"],
          dependencies=[Depends(_check_api_key)])
def submit_reconstruction(req: ReconstructRequest, background_tasks: BackgroundTasks):
    """
    Submit a reconstruction job.

    Runs the trained SR model over all `.npy` patches in `input_dir`,
    generates spectral plots, and uploads everything to R2.
    All paths accept `r2://key` references.
    """
    job = _new_job("reconstruct", _safe_request_dump(req))
    background_tasks.add_task(_run_reconstruction, job["job_id"], req)
    logger.info("Queued reconstruction job %s", job["job_id"])
    return JobStatusResponse(**job)


@app.get("/jobs", response_model=List[JobStatusResponse], tags=["jobs"],
         dependencies=[Depends(_check_api_key)])
def list_jobs(kind: Optional[str] = None, status_filter: Optional[str] = None):
    """List jobs, optionally filtered by `kind` (train/reconstruct) or `status`."""
    jobs = list(_jobs.values())
    if kind:
        jobs = [j for j in jobs if j["kind"] == kind]
    if status_filter:
        jobs = [j for j in jobs if j["status"] == status_filter]
    return [JobStatusResponse(**j) for j in sorted(jobs, key=lambda j: j["created_at"])]


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"],
         dependencies=[Depends(_check_api_key)])
def get_job(job_id: str):
    """Get status and results for a specific job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JobStatusResponse(**_jobs[job_id])


@app.get("/experiments", tags=["storage"], dependencies=[Depends(_check_api_key)])
def get_experiments(
    r2_access_key_id: Optional[str] = None,
    r2_secret_access_key: Optional[str] = None,
    r2_endpoint_url: Optional[str] = None,
    r2_experiments_bucket: Optional[str] = None,
):
    """
    List top-level experiment folders in the experiments R2 bucket.

    Credentials can be supplied as query parameters or will fall back
    to environment variables / .env file.
    """
    try:
        cfg = R2Config.from_env()
        if r2_access_key_id:
            cfg.access_key_id = r2_access_key_id
        if r2_secret_access_key:
            cfg.secret_access_key = r2_secret_access_key
        if r2_endpoint_url:
            cfg.endpoint_url = r2_endpoint_url
        if r2_experiments_bucket:
            cfg.experiments_bucket = r2_experiments_bucket
        experiments = list_experiments(cfg=cfg)
        return {"experiments": experiments, "count": len(experiments),
                "bucket": cfg.experiments_bucket}
    except Exception as exc:
        raise HTTPException(status_code=503,
                            detail=f"Could not reach R2: {exc}") from exc


# ---------------------------------------------------------------------------
# Server entrypoint
# ---------------------------------------------------------------------------

def start_server():
    """Start uvicorn server (called via `spectral-sr-api` console script)."""
    import uvicorn
    uvicorn.run(
        "spectral_sr.api:app",
        host=env.getenv(env.HOST, "0.0.0.0"),
        port=int(env.getenv(env.PORT, "8000")),
        workers=int(env.getenv(env.WORKERS, "1")),
        log_level=env.getenv(env.LOG_LEVEL, "info"),
    )


if __name__ == "__main__":
    start_server()
