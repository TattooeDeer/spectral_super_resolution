# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – builder
# Install UV and build the wheel in an isolated layer so the final image is
# smaller and doesn't contain build tools.
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

WORKDIR /build

# Copy dependency manifest first (layer-cache friendly)
COPY pyproject.toml .
COPY src/ src/

# Create venv and install all dependencies into it
RUN uv venv /opt/venv --python python3
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install the package and its dependencies
RUN uv pip install -e .

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS runtime

# Non-root user for security (RunPod typically runs as root, adjust if needed)
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && useradd -u ${UID} -g ${GID} -m appuser || true

# Runtime system deps (libGL for matplotlib headless rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the populated venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source
WORKDIR /app
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser pyproject.toml .

# Re-install in editable mode inside the runtime stage so entry points work
RUN pip install --no-deps -e .

# Matplotlib headless backend
ENV MPLBACKEND=Agg

# Default env vars (override at runtime via --env or RunPod secrets)
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=1
ENV LOG_LEVEL=info

# R2 credentials – MUST be supplied at runtime, never baked into image
ENV R2_ENDPOINT_URL=https://db79fd90f3dfd68702afa1f74d455523.r2.cloudflarestorage.com
ENV R2_BUCKET_NAME=spectral-reconstruction-experiments
# ENV R2_ACCESS_KEY_ID=     <-- set at runtime
# ENV R2_SECRET_ACCESS_KEY= <-- set at runtime
# ENV API_KEY=              <-- set at runtime (leave empty to disable auth)

# Data volume mount point (mount your RunPod network volume here)
VOLUME ["/data"]

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

USER appuser

CMD ["spectral-sr-api"]
