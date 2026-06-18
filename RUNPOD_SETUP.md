# RunPod + Cloudflare R2 Setup Guide

Complete walkthrough for deploying the Spectral SR API on a RunPod GPU pod,
with all data and model artefacts stored in Cloudflare R2.

The project data lives in the **`spectral-reconstruction-data-ena`** bucket.
Experiment outputs are written to **`spectral-reconstruction-experiments`**.
Objects sit at the **bucket root** (there is no `data/` prefix).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Cloudflare R2 – Create bucket and API token](#2-cloudflare-r2--create-bucket-and-api-token)
3. [R2 bucket layout](#3-r2-bucket-layout)
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
| [GitHub CLI](https://cli.github.com/) | Authenticate with GHCR locally | `brew install gh` |
| GitHub PAT (`GHCR_TOKEN`) | Pull private image on RunPod | See [§4c](#4c-create-a-github-pat-ghcr_token) |
| [rclone](https://rclone.org/) or [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) | Browse/upload data in R2 | See §3 |
| A [RunPod](https://runpod.io) account | GPU compute | Sign up at runpod.io |
| A [Cloudflare](https://dash.cloudflare.com) account | R2 object storage | Free tier available |

---

## 2. Cloudflare R2 – Create bucket and API token

### 2a. Use the existing bucket

The project dataset is already stored in **`spectral-reconstruction-data-ena`**.

If you are setting up a fresh Cloudflare account instead:

1. Log in to **dash.cloudflare.com**
2. Go to **R2 Object Storage** in the left sidebar
3. Click **Create bucket** → name it `spectral-reconstruction-data-ena`
4. Upload the dataset (see §3) or copy it from the shared bucket

> Note your **Account ID** from the URL bar:
> `https://dash.cloudflare.com/<ACCOUNT_ID>/r2`

### 2b. Create an API token

1. Inside R2, click **Manage R2 API Tokens** → **Create API token**
2. Set permissions:
   - **Permissions:** Object Read & Write
   - **Specify bucket(s):** `spectral-reconstruction-data-ena`, `spectral-reconstruction-experiments`
3. Click **Create API token**
4. **Save the token values immediately** – you cannot view the secret again:
   ```
   Access Key ID:     <SPECTRAL_RECONSTRUCTION_R2_ACCESS_KEY_ID>
   Secret Access Key: <SPECTRAL_RECONSTRUCTION_R2_SECRET_ACCESS_KEY>
   ```

### 2c. Note your endpoint URL

Your R2 S3-compatible endpoint is:
```
https://<ACCOUNT_ID>.r2.cloudflarestorage.com
```

---

## 3. R2 bucket layout

### Current bucket structure

```
spectral-reconstruction-data-ena/
  hyperion_train_npy/        ← 64×64×175 training patches  (use for training)
  hyperion_test_npy/         ← 64×64×175 test patches
  landsat_train_npy/         ← 64×64×7 training patches
  landsat_test_npy/          ← 64×64×7 test patches
  Hyperion_train_set/      ← full-scene .npy files (1.npy … 10.npy) — preprocessing source
  Hyperion_test_set/         ← full-scene Hyperion test images
  LandSat_train_set/         ← full-scene Landsat training images
  LandSat_test_set/          ← full-scene Landsat test images
  Models/                    ← pre-trained checkpoints (.pt)
  Raw Images/                ← original source imagery
  ALI/                       ← ALI sensor data
  ARAD_1k/                   ← ARAD dataset
  Manuales/                  ← documentation
  Deprecated/                ← archived files
  ALI_envi_ascii.txt         ← band response metadata (Hyperion / ALI)
  datasets.py                ← original dataset loader (reference)
  *.ipynb                    ← preprocessing / experiment notebooks

spectral-reconstruction-experiments/
  <experiment_name>/         ← outputs written here by training/reconstruction jobs
    <timestamp>/
      config.json
      checkpoints/
      plots/
      metrics.json
```

### Paths used by the API / CLI

| Purpose | R2 key (no `data/` prefix) |
|---------|---------------------------|
| Hyperion train patches | `r2://hyperion_train_npy` |
| Hyperion test patches | `r2://hyperion_test_npy` |
| Landsat train patches | `r2://landsat_train_npy` |
| Landsat test patches | `r2://landsat_test_npy` |
| Pre-trained autoencoder | `r2://Models/AEHG_150_100_75.pt` |
| Band metadata (plots) | `r2://ALI_envi_ascii.txt` |

> **Validation splits:** The trainer requires `hyperion_val_npy` and `landsat_val_npy`,
> which are **not** in the bucket yet. Hold out ~10 % of patches from each train folder,
> upload them as `hyperion_val_npy/` and `landsat_val_npy/` at the bucket root, then
> reference them as `r2://hyperion_val_npy` and `r2://landsat_val_npy`.
> For a quick smoke test only, you may temporarily point val dirs at the test folders.

### Browse or sync with rclone

```bash
# Install rclone
brew install rclone   # macOS

# Configure an R2 remote (run once)
rclone config create r2 s3 \
  provider=Cloudflare \
  access_key_id=<SPECTRAL_RECONSTRUCTION_R2_ACCESS_KEY_ID> \
  secret_access_key=<SPECTRAL_RECONSTRUCTION_R2_SECRET_ACCESS_KEY> \
  endpoint=https://<ACCOUNT_ID>.r2.cloudflarestorage.com

# List top-level contents
rclone ls r2:spectral-reconstruction-data-ena --dirs-only

# List patch files in a training set
rclone ls r2:spectral-reconstruction-data-ena/hyperion_train_npy/

# Download validation splits you created locally
rclone copy /local/path/to/hyperion_val_npy \
  r2:spectral-reconstruction-data-ena/hyperion_val_npy --progress
rclone copy /local/path/to/landsat_val_npy \
  r2:spectral-reconstruction-data-ena/landsat_val_npy --progress
```

### Option B – AWS CLI

```bash
aws configure set aws_access_key_id     <SPECTRAL_RECONSTRUCTION_R2_ACCESS_KEY_ID>
aws configure set aws_secret_access_key <SPECTRAL_RECONSTRUCTION_R2_SECRET_ACCESS_KEY>

ENDPOINT=https://<ACCOUNT_ID>.r2.cloudflarestorage.com

aws s3 ls s3://spectral-reconstruction-data-ena/ --endpoint-url $ENDPOINT
aws s3 sync /local/path/to/hyperion_val_npy \
  s3://spectral-reconstruction-data-ena/hyperion_val_npy \
  --endpoint-url $ENDPOINT
```

---

## 4. Build the Docker image and push to GHCR

### Option A – Let GitHub Actions build it (easiest)

`.github/workflows/docker.yml` publishes images from **`master`** and **release tags** only.
Pushes to `sandbox` do not trigger a build — open a PR to `master` to validate the Dockerfile,
then merge to publish.

| Trigger | Builds? | Pushes to GHCR? | Example tags |
|---------|---------|-------------------|--------------|
| PR → `master` | Yes | No | — |
| Push to `master` | Yes | Yes | `:latest`, `:dev-<sha>`, `:sha-<sha>` |
| Push tag `v1.0.0` | Yes | Yes | `:1.0.0`, `:1.0` |

**Release workflow (recommended for RunPod):**

1. Merge your changes into `master` via pull request
2. Tag the release commit: `git tag v1.0.0 && git push origin v1.0.0`
3. Watch the build at: `github.com/<user>/spectral_super_resolution/actions`
4. Deploy with: `ghcr.io/<user>/spectral_super_resolution:1.0.0`

For bleeding-edge deploys you can use `:latest` after a `master` push, but pinned semver
tags (e.g. `:1.0.0`) are safer for production pods.

The workflow uses the built-in `GITHUB_TOKEN` to push to GHCR; you do not need
to create a `GHCR_TOKEN` secret for CI unless you customize the workflow.
See [§5c](#5c-github-actions-auto-build-no-extra-trigger) for billing notes.

### Option B – Build locally and push manually

You need a GitHub Personal Access Token (PAT) with **`write:packages`** (and **`read:packages`**) to push.
See [§4c](#4c-create-a-github-pat-ghcr_token) to create one, then:

```bash
# Export the token (often called GHCR_TOKEN or GITHUB_TOKEN)
export GHCR_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx

# Authenticate with GHCR (username = your GitHub username, all lowercase)
echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-username> --password-stdin

# Build (takes ~5 min first time due to PyTorch download)
docker build -t ghcr.io/<github-username>/spectral_super_resolution:1.0.0 .

# Push
docker push ghcr.io/<github-username>/spectral_super_resolution:1.0.0
```

### 4c. Create a GitHub PAT (`GHCR_TOKEN`)

A **GitHub Personal Access Token** is what people usually store as `GHCR_TOKEN`.
It is **not** a separate GitHub product — it is a PAT scoped for `ghcr.io`.

#### Classic token (simplest)

1. Open **[github.com/settings/tokens](https://github.com/settings/tokens)** → **Generate new token** → **Generate new token (classic)**
2. Give it a name, e.g. `runpod-ghcr-pull`
3. Select scopes:
   | Use case | Required scopes |
   |----------|-----------------|
   | **Pull** image on RunPod (private package) | `read:packages` |
   | **Push** image from your machine | `write:packages` (includes read) |
   | Package owned by an **org** | Also add `read:org` if pull fails with 403 |
4. Click **Generate token** and copy the value (`ghp_…` or `github_pat_…`).
   **You cannot view it again.**

#### Fine-grained token (alternative)

1. Open **[github.com/settings/personal-access-tokens](https://github.com/settings/personal-access-tokens)** → **Generate new token**
2. Set **Repository access** to the `spectral_super_resolution` repo (or your org)
3. Under **Permissions → Packages**, set **Read** (pull) or **Read and write** (push)
4. Generate and copy the token

#### Verify the token works

```bash
export GHCR_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx

# Should print "Login Succeeded"
echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-username> --password-stdin

# Should download layers (public or private, depending on package visibility)
docker pull ghcr.io/<github-username>/spectral_super_resolution:1.0.0
```

> **Security:** Treat `GHCR_TOKEN` like a password. Store it in a password manager,
> GitHub Actions secrets, or RunPod Container Registry Auth — never commit it to git.

---

## 5. Deploy on RunPod

RunPod must authenticate to `ghcr.io` when your image is **private**.
Choose one of the two approaches below.

### 5a. Option A – Public image (no token needed)

Simplest if you are fine with the image being publicly pullable:

1. Go to **github.com/<user>/spectral_super_resolution/pkgs/container/spectral_super_resolution**
2. Click **Package settings** → **Change visibility** → **Public**
3. Deploy the pod (§5d) — leave **Container Registry Auth** empty

### 5b. Option B – Private image (GHCR token via Container Registry Auth)

Keep the package private and let RunPod pull it with your PAT.

#### Step 1 – Create a pull-only PAT

Follow [§4c](#4c-create-a-github-pat-ghcr_token) with at least the **`read:packages`** scope.
This is your `GHCR_TOKEN`.

#### Step 2 – Save credentials in RunPod

**Via the web console:**

1. Log in to **[runpod.io](https://runpod.io)** → click your profile (top right) → **Settings**
2. Open **Container Registry Auth** (sometimes listed under **Connections**)
3. Click **Create** / **Add credentials**
4. Fill in:

   | Field | Value |
   |-------|-------|
   | Name | `ghcr-spectral-sr` (any label you like) |
   | Username | Your **GitHub username in lowercase** (not your email) |
   | Password | Your `GHCR_TOKEN` / PAT (`ghp_…`) |

5. Save. RunPod stores the credential and shows it in the list.

**Via CLI (`runpodctl`):**

```bash
runpodctl registry create \
  --name "ghcr-spectral-sr" \
  --username "<github-username>" \
  --password "$GHCR_TOKEN"
```

**Via REST API:**

```bash
curl -X POST https://rest.runpod.io/v1/containerregistryauth \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ghcr-spectral-sr",
    "username": "<github-username>",
    "password": "'"$GHCR_TOKEN"'"
  }'
```

Note the returned `id` — that is your `containerRegistryAuthId`.

#### Step 3 – Attach credentials when deploying

When creating or editing a **Pod template** or deploying a **Pod**:

1. Set **Container image** to `ghcr.io/<github-username>/spectral_super_resolution:1.0.0`
2. In **Registry credentials** / **Container Registry Auth**, select **`ghcr-spectral-sr`**
   (the entry you created in Step 2)
3. Continue with the rest of the pod settings in §5d

> **Important:** RunPod uses **Container Registry Auth** to pull the image at startup.
> Do **not** put `GHCR_TOKEN` in the pod's environment variables — that does not
> authenticate the image pull. Env vars are only for your app (R2, SPECTRAL_RECONSTRUCTION_API_KEY, etc.).

### 5c. GitHub Actions auto-build (no extra trigger)

The workflow runs on **pull requests to `master`** (build only) and on **pushes to `master`**
or **version tags** (`v*.*.*`) to publish images. You do **not** need a separate webhook
or trigger in the GitHub UI.

| Trigger | Publishes to GHCR? |
|---------|-------------------|
| PR → `master` | No (validates Dockerfile only) |
| Push to `master` | Yes — `:latest`, `:dev-<sha>`, `:sha-<sha>` |
| Push tag `v1.0.0` | Yes — `:1.0.0`, `:1.0` |

| Repo visibility | Actions cost |
|---------------|--------------|
| **Public** | Unlimited free minutes |
| **Private** | Free tier includes ~2 000 min/month; each Docker build is ~5–15 min |

Watch builds at **github.com/<user>/spectral_super_resolution/actions**.
For RunPod, prefer a pinned release tag such as `:1.0.0`.

### 5d. Create a new pod

1. Log in to **runpod.io** → **Pods** → **+ Deploy**
2. Select a GPU (A4000 or RTX3090 is sufficient; A100 for faster training)
3. Click **Edit Template** and fill in:

| Field | Value |
|-------|-------|
| Container image | `ghcr.io/<github-username>/spectral_super_resolution:1.0.0` |
| Container Registry Auth | `ghcr-spectral-sr` *(§5b — skip if image is public)* |
| Container disk | 20 GB |
| Expose HTTP ports | `8000` |
| Volume mount path | `/data` (if you use a network volume) |

4. Under **Environment Variables**, add:

```
SPECTRAL_RECONSTRUCTION_R2_ACCESS_KEY_ID              = <your value>
SPECTRAL_RECONSTRUCTION_R2_SECRET_ACCESS_KEY          = <your value>
SPECTRAL_RECONSTRUCTION_R2_ENDPOINT_URL               = https://<ACCOUNT_ID>.r2.cloudflarestorage.com
SPECTRAL_RECONSTRUCTION_R2_BUCKET_NAME                = spectral-reconstruction-data-ena
SPECTRAL_RECONSTRUCTION_R2_BUCKET_NAME_EXPERIMENTS    = spectral-reconstruction-experiments
SPECTRAL_RECONSTRUCTION_API_KEY                       = <a strong random string>
SPECTRAL_RECONSTRUCTION_PORT                          = 8000
```

> **Tip:** Store R2 and API secrets in **RunPod Secrets** (Settings → Secrets)
> and reference them when deploying, so they are encrypted at rest.

5. Click **Deploy**. The pod will pull the image and start. This takes ~2–3 min.
   If the pull fails, see the GHCR row in [§10 Troubleshooting](#10-troubleshooting).

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

All examples use `curl` with a bearer token. Replace `$SPECTRAL_RECONSTRUCTION_API_KEY` and `$POD_URL`
with your actual values.

```bash
POD_URL=https://abc123-8000.proxy.runpod.net
SPECTRAL_RECONSTRUCTION_API_KEY=your_api_key_here
AUTH="Authorization: Bearer $SPECTRAL_RECONSTRUCTION_API_KEY"
```

### Step 1 – Train the autoencoder

```bash
curl -s -X POST "$POD_URL/train" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{
    "mode": "autoencoder",
    "model": "hourglass",
    "loss": "mse",
    "hyperion_train_dir": "r2://hyperion_train_npy",
    "hyperion_val_dir":   "r2://hyperion_val_npy",
    "encoder_channels": [150, 100, 75],
    "epochs": 5,
    "lr": 0.001,
    "batch_size": 32,
    "experiment_name": "autoencoder_150_100_75",
    "r2": {
      "access_key_id":     "'$SPECTRAL_RECONSTRUCTION_R2_ACCESS_KEY_ID'",
      "secret_access_key": "'$SPECTRAL_RECONSTRUCTION_R2_SECRET_ACCESS_KEY'",
      "endpoint_url":      "https://<ACCOUNT_ID>.r2.cloudflarestorage.com",
      "bucket":            "spectral-reconstruction-data-ena"
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
    "hyperion_train_dir":  "r2://hyperion_train_npy",
    "hyperion_val_dir":    "r2://hyperion_val_npy",
    "landsat_train_dir":   "r2://landsat_train_npy",
    "landsat_val_dir":     "r2://landsat_val_npy",
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
    "input_dir":                   "r2://landsat_test_npy",
    "ground_truth_dir":            "r2://hyperion_test_npy",
    "hyperion_properties_path":    "r2://ALI_envi_ascii.txt",
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
  --hyperion-train-dir r2://hyperion_train_npy \
  --hyperion-val-dir   r2://hyperion_val_npy   \
  --encoder-channels 150,100,75 --epochs 5 \
  --output-dir ./outputs/ae \
  --r2-upload --r2-experiment-name autoencoder_150_100_75

# Train SR with perceptual loss, AE checkpoint from R2
spectral-sr train \
  --mode sr --loss perceptual \
  --ae-checkpoint      r2://Models/AEHG_150_100_75.pt \
  --hyperion-train-dir r2://hyperion_train_npy   \
  --hyperion-val-dir   r2://hyperion_val_npy     \
  --landsat-train-dir  r2://landsat_train_npy    \
  --landsat-val-dir    r2://landsat_val_npy      \
  --encoder-channels 150,100,75 --epochs 5 \
  --output-dir ./outputs/sr \
  --r2-upload --r2-experiment-name sr_perceptual

# Reconstruct from R2 model and data
spectral-sr reconstruct \
  --model hourglass \
  --checkpoint   r2://Models/hg_SR_150_100_75_perceptual.pt \
  --input-dir    r2://landsat_test_npy \
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
| `No .npy files found` | Wrong path or empty download | Verify R2 keys with `rclone ls r2:spectral-reconstruction-data-ena/hyperion_train_npy/` |
| `hyperion_val_dir does not exist` | Val splits not uploaded | Create and upload `hyperion_val_npy/` and `landsat_val_npy/` (see §3) |
| API returns 401 | `SPECTRAL_RECONSTRUCTION_API_KEY` mismatch | Check `Authorization: Bearer <key>` header |
| Container fails to start | Port conflict | Set `SPECTRAL_RECONSTRUCTION_PORT=8000` in RunPod env vars |
| `unauthorized` / `denied` pulling image | Private GHCR package without registry auth | Create PAT (§4c), add Container Registry Auth (§5b), select it on the pod |
| `unauthorized` pulling public image | Stale local `docker login` on your machine | Run `docker logout ghcr.io` and retry |
| GHCR 403 with org package | PAT missing org/package access | Add `read:org` scope or grant package read on the org |
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
