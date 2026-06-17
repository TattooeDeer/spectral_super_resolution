# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – builder
# Install UV and resolve dependencies from the frozen lockfile into a venv.
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS builder

# Install system build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency manifests first (layer-cache friendly)
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Create venv and install locked dependencies (including this package, editable)
RUN uv venv /opt/venv --python python3
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv sync --frozen

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS runtime

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

# Copy the populated venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source at the same path as the builder so the editable install resolves
WORKDIR /app
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
COPY --chown=appuser:appuser src/ src/

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
