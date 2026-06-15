# Docker image for the Python stages of the SmartSDG pipeline:
#   OD training, Active Learning, Continual Learning, Inference.
#
# NOT for SDG — Isaac Sim runs natively (see docs/PORTABILITY.md).
#
# Build:  docker build -t smartsdg-train -f docker/Dockerfile .
# Run:    docker run --gpus all -v "$PWD:/workspace" -w /workspace \
#             smartsdg-train python OD_training/main.py --mode train --aas_file <cfg>
#
# Requires the NVIDIA Container Toolkit on the host (Linux, or Windows+WSL2).

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV / Ultralytics runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps first for layer caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Source is mounted at runtime (-v "$PWD:/workspace"), so nothing is COPYed here —
# the image stays small and always uses your current code + data.

CMD ["python", "-c", "import torch, ultralytics; print('SmartSDG train image ready — torch', torch.__version__, '| cuda', torch.cuda.is_available())"]
