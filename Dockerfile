# Single-stage image: pytorch base already ships torch/CUDA in conda.
# uv sync skips those packages to avoid ~3GB duplicate installs that exhaust
# GitHub Actions runner disk during multi-stage COPY.
# CUDA 12.8 + PyTorch 2.7+ required for Blackwell GPUs (sm_120, e.g. RTX PRO 4000).
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# Bake release version into the image (override: docker build --build-arg APP_VERSION=1.2.3)
ARG APP_VERSION=dev
ENV SPECTRAL_RECONSTRUCTION_SPECTRAL_SR_VERSION=${APP_VERSION}
LABEL org.opencontainers.image.version="${APP_VERSION}"

# Non-root user for security (RunPod typically runs as root, adjust if needed)
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && useradd -u ${UID} -g ${GID} -m appuser || true

# Runtime system deps (curl for healthcheck; libGL for matplotlib headless rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_LINK_MODE=copy

WORKDIR /app

# Copy dependency manifests first (layer-cache friendly)
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

# Use conda site-packages for torch; install only app + non-torch deps into venv.
RUN uv venv /opt/venv --python python3 --system-site-packages
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Packages already provided by pytorch/pytorch (see uv.lock for the full CUDA stack).
RUN set -eux; \
    EXCLUDE_PKGS="torch triton cuda-bindings cuda-pathfinder cuda-toolkit \
        nvidia-cublas nvidia-cublas-cu12 \
        nvidia-cuda-cupti nvidia-cuda-cupti-cu12 \
        nvidia-cuda-nvrtc nvidia-cuda-nvrtc-cu12 \
        nvidia-cuda-runtime nvidia-cuda-runtime-cu12 \
        nvidia-cudnn-cu12 nvidia-cudnn-cu13 \
        nvidia-cufft nvidia-cufft-cu12 \
        nvidia-cufile nvidia-cufile-cu12 \
        nvidia-curand nvidia-curand-cu12 \
        nvidia-cusolver nvidia-cusolver-cu12 \
        nvidia-cusparse nvidia-cusparse-cu12 \
        nvidia-cusparselt-cu12 nvidia-cusparselt-cu13 \
        nvidia-nccl-cu12 nvidia-nccl-cu13 \
        nvidia-nvjitlink nvidia-nvjitlink-cu12 \
        nvidia-nvshmem-cu13 \
        nvidia-nvtx nvidia-nvtx-cu12"; \
    ARGS=""; \
    for pkg in ${EXCLUDE_PKGS}; do ARGS="${ARGS} --no-install-package ${pkg}"; done; \
    uv sync --frozen ${ARGS}; \
    rm -rf /root/.cache/uv

RUN chown -R appuser:appuser /opt/venv /app

# Matplotlib headless backend
ENV MPLBACKEND=Agg

# Default env vars (override at runtime via --env or RunPod secrets)
ENV SPECTRAL_RECONSTRUCTION_HOST=0.0.0.0
ENV SPECTRAL_RECONSTRUCTION_PORT=8000
ENV SPECTRAL_RECONSTRUCTION_WORKERS=1
ENV SPECTRAL_RECONSTRUCTION_LOG_LEVEL=info

# R2 credentials – MUST be supplied at runtime (see .env.example / RUNPOD_SETUP.md)
# ENV SPECTRAL_RECONSTRUCTION_R2_ACCESS_KEY_ID=
# ENV SPECTRAL_RECONSTRUCTION_R2_SECRET_ACCESS_KEY=
# ENV SPECTRAL_RECONSTRUCTION_R2_ENDPOINT_URL=
# ENV SPECTRAL_RECONSTRUCTION_R2_BUCKET_NAME=
# ENV SPECTRAL_RECONSTRUCTION_R2_BUCKET_NAME_EXPERIMENTS=
# ENV SPECTRAL_RECONSTRUCTION_API_KEY=

# Data volume mount point (mount your RunPod network volume here)
VOLUME ["/data"]

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${SPECTRAL_RECONSTRUCTION_PORT}/health || exit 1

USER appuser

CMD ["spectral-sr-api"]
