# syntax=docker/dockerfile:1.6
# =============================================================
# opencv-image-edit — production Docker image
# Multi-stage build: download model files in stage 1, ship a
# lean runtime image in stage 2. No torch, no rembg, no
# Real-ESRGAN — just opencv-contrib-python-headless + onnxruntime.
# Target image size: ~800 MB.
# =============================================================

# =============================================================
# Stage 1: Model downloader
# Downloads AI model files (U2NetP ONNX, EDSR .pb) so the app
# container can use them without re-downloading at runtime.
# =============================================================
FROM python:3.12-slim AS model-downloader

WORKDIR /dl

# ca-certificates so HTTPS downloads to github.com work
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The download script uses only the Python stdlib (urllib, hashlib,
# pathlib, os, sys) — no extra pip packages needed.
COPY scripts/download_models.py /dl/download_models.py

# Run the downloader; output goes to /models
RUN python /dl/download_models.py /models


# =============================================================
# Stage 2: Application
# =============================================================
FROM python:3.12-slim AS app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV needs libgl1 + libglib2.0-0 at runtime.
# DejaVu is used by comparison.py (text labels on before/after).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        fonts-dejavu-core \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/app

# Install Python deps first for layer caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# Copy application code
COPY app/ /home/app/app/
COPY web/ /home/app/web/
COPY scripts/ /home/app/scripts/

# Copy pre-downloaded models from stage 1
COPY --from=model-downloader /models /home/app/models

# Non-root user
RUN useradd --create-home --uid 1000 --shell /bin/bash appuser \
    && mkdir -p /tmp/opencv-image-edit \
    && chown -R appuser:appuser /home/app /tmp/opencv-image-edit
USER appuser

# Default config (overridable via env / docker-compose)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    ENABLE_METRICS=true \
    METRICS_PORT=9090 \
    MODEL_DIR=/home/app/models \
    LOG_LEVEL=INFO

EXPOSE 8000 9090

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, sys; r = urllib.request.urlopen('http://localhost:8000/health', timeout=3); sys.exit(0 if r.getcode() == 200 else 1)"

CMD ["python", "-m", "app.main"]
