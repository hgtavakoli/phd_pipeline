# SmartSDG Pipeline v2 — Docker image
#
# Covers: OD Training · Active Learning · Continual Learning · Inference · Dashboard
# NOT for SDG (Isaac Sim runs natively — see docs/PORTABILITY.md)
#
# Build:
#   docker build -t smartsdg -f docker/Dockerfile .
#
# Quick-start (dashboard, GPU host):
#   docker compose -f docker/docker-compose.yml up dashboard
#
# One-off training run:
#   docker run --gpus all -v "$PWD:/workspace" smartsdg train --aas_file AAS/datasets/.../foo.json
#
# CPU-only (no GPU on host):
#   docker run -v "$PWD:/workspace" smartsdg train --aas_file ...
#
# GPU requires NVIDIA Container Toolkit on Linux / Windows+WSL2.

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Prevent ultralytics from writing to /root outside the workspace
    YOLO_CONFIG_DIR=/workspace/.ultralytics

# System runtime deps: OpenCV, Ultralytics, libcamera (headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps (cached layer — only re-runs when requirements.txt changes)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

# Copy entrypoint — source code is bind-mounted at runtime, not baked in
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8765

ENTRYPOINT ["/entrypoint.sh"]

# Default: print env info (override with e.g. `train`, `dashboard`, `shell`)
CMD ["info"]
