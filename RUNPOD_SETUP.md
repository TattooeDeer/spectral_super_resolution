# RunPod + Cloudflare R2 Setup Guide

Complete walkthrough for deploying the Spectral SR API on a RunPod GPU pod,
with all data and model artefacts stored in Cloudflare R2.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Cloudflare R2 – Create bucket and API token](#2-cloudflare-r2--create-bucket-and-api-token)
3. [Upload your data to R2](#3-upload-your-data-to-r2)
4. [Build the Docker image and push to GHCR](#4-build-the-docker-image-and-push-to-ghcr)
5. [Deploy on RunPod](#5-deploy-on-runpod)
6. [Configure the API server](#6-configure-the-api-server)
7. [Run your first experiment](#7-run-your-first-experiment)
8. [Monitor jobs and retrieve results](#8-monitor-jobs-and-retrieve-results)
9. [CLI usage (alternative to API)](#9-cli-usage-alternative-to-api)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

| Tool | Why | Install |
|------|-----|---------|
| [Docker](https://docs.docker.com/get-docker/) | Build/test image locally | `brew install --cask docker` |
| [GitHub CLI](https://cli.github.com/) | Authenticate with GHCR | `brew install gh` |
| [rclone](https://rclone.org/) or [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) | Upload data to R2 | See §3 |
| A [RunPod](https://runpod.io) account | GPU compute | Sign up at runpod.io |
| A [Cloudflare](https://dash.cloudflare.com) account | R2 object storage | Free tier available |

---

## 2. Cloudflare R2 – Create bucket and API token

### 2a. Create the bucket

1. Log in to **dash.cloudflare.com**
2. Go to **R2 Object Storage** in the left sidebar
3. Click **Create bucket** → name it `spectral-reconstruction-experiments`
4. Leave all defaults and click **Create bucket**

> Note your **Account ID** from the URL bar:
> `https://dash.cloudflare.com/<ACCOUNT_ID>/r2`

### 2b. Create an API token

1. Inside R2, click **Manage R2 API Tokens** → **Create API token**
2. Set permissions:
   - **Permissions:** Object Read & Write
   - **Specify bucket(s):** `spectral-reconstruction-experiments`
3. Click **Create API token**
4. **Save the token values immediately** – you cannot view the secret again:
   ```
   Access Key ID:     <R2_ACCESS_KEY_ID>
   Secret Access Key: <R2_SECRET_ACCESS_KEY>
   ```

### 2c. Note your endpoint URL

Your R2 S3-compatible endpoint is:
```
https://<ACCOUNT_ID>.r2.cloudflarestorage.com
```

---

## 3. Upload your data to R2

R2 is S3-compatible. Use **rclone** or the **AWS CLI** pointed at the R2 endpoint.

### Option A – rclone (recommended)

```bash
# Install rclone
brew install rclone   # macOS

# Configure an R2 remote (run once)
rclone config create r2 s3 \
  provider=Cloudflare \
  access_key_id=<R2_ACCESS_KEY_ID> \
  secret_access_key=<R2_SECRET_ACCESS_KEY> \
  endpoint=https://<ACCOUNT_ID>.r2.cloudflarestorage.com

# Upload a directory (e.g., Hyperion train patches)
rclone copy /local/path/to/hyperion_train_npy \
  r2:spectral-reconstruction-experiments/data/hyperion_train_npy \
  --progress

# Upload all dataset directories
for DIR in hyperion_train_npy hyperion_val_npy hyperion_test_npy \
           landsat_train_npy  landsat_val_npy  landsat_test_npy; do
  rclone copy /local/path/to/$DIR \
    r2:spectral-reconstruction-experiments/data/$DIR --progress
done

# Upload metadata files
rclone copy /local/path/to/propertiesH_metadata.txt \
  r2:spectral-reconstruction-experiments/data/
rclone copy /local/path/to/propertiesL8_metadata.txt \
  r2:spectral-reconstruction-experiments/data/

# Upload a pre-trained model checkpoint (if you have one)
rclone copy /local/path/to/AEHG_150_100_75.pt \
  r2:spectral-reconstruction-experiments/models/
```

### Option B – AWS CLI

```bash
# Configure AWS CLI to point at R2
aws configure set aws_access_key_id     <R2_ACCESS_KEY_ID>
aws configure set aws_secret_access_key <R2_SECRET_ACCESS_KEY>

ENDPOINT=https://<ACCOUNT_ID>.r2.cloudflarestorage.com

aws s3 sync /local/path/to/hyperion_train_npy \
  s3://spectral-reconstruction-experiments/data/hyperion_train_npy \
  --endpoint-url $ENDPOINT

# Repeat for all directories…
```

### Suggested R2 bucket layout

```
spectral-reconstruction-experiments/
  data/
    hyperion_train_npy/      ← .npy patches  (64×64×175)
    hyperion_val_npy/
    hyperion_test_npy/
    landsat_train_npy/       ← .npy patches  (64×64×7)
    landsat_val_npy/
    landsat_test_npy/
    propertiesH_metadata.txt
    propertiesL8_metadata.txt
  models/
    AEHG_150_100_75.pt       ← pre-trained autoencoder (optional)
  <experiment_name>/
    <timestamp>/
      config.json
      checkpoints/
      plots/
      metrics.json
```

---

## 4. Build the Docker image and push to GHCR

### Option A – Let GitHub Actions build it (easiest)

Every push to the `sandbox` branch automatically builds and pushes
the image to `ghcr.io/<your-github-username>/spectral_super_resolution`.

1. Push your changes: `git push origin sandbox`
2. Watch the build at: `github.com/<user>/spectral_super_resolution/actions`
3. Once green, the image is at: `ghcr.io/<user>/spectral_super_resolution:sandbox`

### Option B – Build locally and push manually

```bash
# Authenticate with GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u <github-username> --password-stdin

# Build (takes ~5 min first time due to PyTorch download)
docker build -t ghcr.io/<github-username>/spectral_super_resolution:sandbox .

# Push
docker push ghcr.io/<github-username>/spectral_super_resolution:sandbox
```

---

## 5. Deploy on RunPod

### 5a. Make the GHCR image public (or configure auth)

1. Go to **github.com/<user>/spectral_super_resolution/pkgs/container/spectral_super_resolution**
2. Click **Package settings** → **Change visibility** → **Public**

*(Alternatively, skip this and supply `GHCR_TOKEN` as a RunPod secret.)*

### 5b. Create a new pod

1. Log in to **runpod.io** → **Pods** → **+ Deploy**
2. Select a GPU (A4000 or RTX3090 is sufficient; A100 for faster training)
3. Click **Edit Template** and fill in:

| Field | Value |
|-------|-------|
| Container image | `ghcr.io/<github-username>/spectral_super_resolution:sandbox` |
| Container disk | 20 GB |
| Expose HTTP ports | `8000` |
| Volume mount path | `/data` (if you use a network volume) |

4. Under **Environment Variables**, add:

```
R2_ACCESS_KEY_ID     = <your value>
R2_SECRET_ACCESS_KEY = <your value>
R2_ENDPOINT_URL      = https://<ACCOUNT_ID>.r2.cloudflarestorage.com
R2_BUCKET_NAME       = spectral-reconstruction-experiments
API_KEY              = <a strong random string>
PORT                 = 8000
```

> **Tip:** Store secrets in **RunPod Secrets** (pod settings → Secrets tab)
> so they are encrypted and injected at runtime.

5. Click **Deploy**. The pod will pull the image and start. This takes ~2–3 min.

---

## 6. Configure the API server

Once the pod is running:

1. Copy the public HTTP URL from the RunPod dashboard (looks like
   `https://abc123-8000.proxy.runpod.net`).

2. Test the health endpoint:
   ```bash
   curl https://abc123-8000.proxy.runpod.net/health
   ```
   Expected response:
   ```json
   {"status":"ok","cuda_available":true,"device":"cuda","device_name":"NVIDIA RTX A4000",...}
   ```

3. Open the interactive API docs at:
   ```
   https://abc123-8000.proxy.runpod.net/docs
   ```

---

## 7. Run your first experiment

All examples use `curl` with a bearer token. Replace `$API_KEY` and `$POD_URL`
with your actual values.

```bash
POD_URL=https://abc123-8000.proxy.runpod.net
API_KEY=your_api_key_here
AUTH="Authorization: Bearer $API_KEY"
```

### Step 1 – Train the autoencoder

```bash
curl -s -X POST "$POD_URL/train" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{
    "mode": "autoencoder",
    "model": "hourglass",
    "loss": "mse",
    "hyperion_train_dir": "r2://data/hyperion_train_npy",
    "hyperion_val_dir":   "r2://data/hyperion_val_npy",
    "encoder_channels": [150, 100, 75],
    "epochs": 5,
    "lr": 0.001,
    "batch_size": 32,
    "experiment_name": "autoencoder_150_100_75",
    "r2": {
      "access_key_id":     "'$R2_ACCESS_KEY_ID'",
      "secret_access_key": "'$R2_SECRET_ACCESS_KEY'",
      "endpoint_url":      "https://<ACCOUNT_ID>.r2.cloudflarestorage.com",
      "bucket":            "spectral-reconstruction-experiments"
    }
  }' | python3 -m json.tool
```

Save the returned `job_id`.

### Step 2 – Poll for completion

```bash
JOB_ID=<job_id from above>

curl -s "$POD_URL/jobs/$JOB_ID" -H "$AUTH" | python3 -m json.tool
```

When `"status": "done"` the `r2_keys` field lists every uploaded file.

### Step 3 – Train SR with perceptual loss

After the autoencoder job completes, note the R2 key of its final checkpoint
(e.g. `autoencoder_150_100_75/20240601_120000/output/checkpoint_epoch_final.pt`).

```bash
curl -s -X POST "$POD_URL/train" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{
    "mode": "sr",
    "model": "hourglass",
    "loss": "perceptual",
    "ae_checkpoint":       "r2://autoencoder_150_100_75/20240601_120000/output/checkpoint_epoch_final.pt",
    "hyperion_train_dir":  "r2://data/hyperion_train_npy",
    "hyperion_val_dir":    "r2://data/hyperion_val_npy",
    "landsat_train_dir":   "r2://data/landsat_train_npy",
    "landsat_val_dir":     "r2://data/landsat_val_npy",
    "encoder_channels": [150, 100, 75],
    "epochs": 5,
    "content_loss_coeff": 1.0,
    "style_loss_coeff": 0.001,
    "experiment_name": "sr_perceptual_150_100_75"
  }' | python3 -m json.tool
```

### Step 4 – Reconstruct and generate plots

```bash
SR_CKPT="r2://sr_perceptual_150_100_75/20240601_130000/output/checkpoint_epoch_final.pt"

curl -s -X POST "$POD_URL/reconstruct" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{
    "model": "hourglass",
    "checkpoint":                  "'$SR_CKPT'",
    "input_dir":                   "r2://data/landsat_test_npy",
    "ground_truth_dir":            "r2://data/hyperion_test_npy",
    "hyperion_properties_path":    "r2://data/propertiesH_metadata.txt",
    "landsat_properties_path":     "r2://data/propertiesL8_metadata.txt",
    "encoder_channels": [150, 100, 75],
    "num_plot_samples": 5,
    "experiment_name": "reconstruction_test"
  }' | python3 -m json.tool
```

---

## 8. Monitor jobs and retrieve results

### List all jobs

```bash
curl -s "$POD_URL/jobs" -H "$AUTH" | python3 -m json.tool
```

### List experiments in R2

```bash
curl -s "$POD_URL/experiments" -H "$AUTH" | python3 -m json.tool
```

### Download results from R2

```bash
# Using rclone (lists everything under an experiment)
rclone ls r2:spectral-reconstruction-experiments/reconstruction_test/

# Download plots
rclone copy r2:spectral-reconstruction-experiments/reconstruction_test/20240601_140000/plots \
  ./local_plots/ --progress

# Download trained model
rclone copy \
  r2:spectral-reconstruction-experiments/sr_perceptual_150_100_75/20240601_130000/output/checkpoint_epoch_final.pt \
  ./models/
```

---

## 9. CLI usage (alternative to API)

If you prefer running training from a terminal (e.g., inside the RunPod pod or on
your local machine after `uv pip install -e .`):

```bash
# All env vars from .env are loaded automatically

# Train autoencoder, pulling data from R2
spectral-sr train \
  --mode autoencoder --model hourglass \
  --hyperion-train-dir r2://data/hyperion_train_npy \
  --hyperion-val-dir   r2://data/hyperion_val_npy   \
  --encoder-channels 150,100,75 --epochs 5 \
  --output-dir ./outputs/ae \
  --r2-upload --r2-experiment-name autoencoder_150_100_75

# Train SR with perceptual loss, AE checkpoint from R2
spectral-sr train \
  --mode sr --loss perceptual \
  --ae-checkpoint      r2://models/AEHG_150_100_75.pt \
  --hyperion-train-dir r2://data/hyperion_train_npy   \
  --hyperion-val-dir   r2://data/hyperion_val_npy     \
  --landsat-train-dir  r2://data/landsat_train_npy    \
  --landsat-val-dir    r2://data/landsat_val_npy      \
  --encoder-channels 150,100,75 --epochs 5 \
  --output-dir ./outputs/sr \
  --r2-upload --r2-experiment-name sr_perceptual

# Reconstruct from R2 model and data
spectral-sr reconstruct \
  --model hourglass \
  --checkpoint   r2://models/hg_SR_150_100_75_perceptual.pt \
  --input-dir    r2://data/landsat_test_npy \
  --output-dir   ./outputs/reconstructed \
  --r2-upload --r2-experiment-name reconstruction_test

# Pass credentials explicitly (overrides .env)
spectral-sr train ... \
  --r2-access-key-id     YOUR_KEY \
  --r2-secret-access-key YOUR_SECRET \
  --r2-endpoint-url      https://<ACCOUNT_ID>.r2.cloudflarestorage.com
```

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `EnvironmentError: Missing R2 credentials` | Env vars not set | Add them to RunPod secrets or your `.env` file |
| `CUDA out of memory` | Batch size too large | Reduce `--batch-size` (try 8 or 16) |
| `No .npy files found` | Wrong path or empty download | Verify R2 keys with `rclone ls r2:bucket/prefix/` |
| API returns 401 | `API_KEY` mismatch | Check `Authorization: Bearer <key>` header |
| Container fails to start | Port conflict | Set `PORT=8000` in RunPod env vars |
| `OMP: Error #179` | OpenMP shared memory | Set `OMP_NUM_THREADS=1` in env vars |
| Slow first request | Model/data still downloading from R2 | Normal – R2 transfers are cached in `/tmp` |
| GitHub Actions build fails | GHCR write permission | Ensure `packages: write` is in the workflow permissions |

### View pod logs on RunPod

In the RunPod dashboard → your pod → **Logs** tab.

### Run interactively inside the container

```bash
# From RunPod dashboard → Connect → Start Terminal
python -c "import spectral_sr; print('Package OK')"
spectral-sr-api  # starts the API server manually
```
