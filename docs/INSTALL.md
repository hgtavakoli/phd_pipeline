# Installation Guide

Two setup paths are available. **Docker is recommended** for any machine except
the one running Isaac Sim — it requires no Python environment management and is
fully reproducible.

```
  SDG (Isaac Sim)            →  native install only (Windows or Linux)
  OD / AL / CL / Inference   →  Docker  ← recommended
                                 OR conda env (if Docker is not available)
```

---

## Hardware requirements

| Component | Minimum |
|---|---|
| GPU | NVIDIA RTX 20-series or later (RTX 2080 8 GB confirmed) |
| RAM | 16 GB |
| Disk | 50 GB free (datasets + model weights) |
| NVIDIA driver | 525+ |

---

## Part A — SDG Setup (Isaac Sim — always native)

Isaac Sim cannot run in Docker on Windows. Install it natively.

**Confirmed working install:** `C:\Users\hota02\Documents\omniverse` (Isaac Sim 4.5.0-rc.36)

### Verify

```bat
C:\Users\hota02\Documents\omniverse\python.bat --version
:: Python 3.10.x

nvidia-smi
:: RTX 2080 Max-Q, CUDA 12.x
```

### Quick SDG test (5 images)

```bat
cd SmartSDG
C:\Users\hota02\Documents\omniverse\python.bat main.py ^
  --config "src\config\config.yaml" ^
  --component_class GlueGun --light_mode Medium ^
  --num_samples 5 --load_env plane ^
  --width 640 --height 480 --convert_to_yolo
```

Output: `SmartSDG\Datasets\yolo_experiments\SINGLE_GlueGun_Medium_plane\`

First run compiles shaders — takes several minutes. Subsequent runs are fast.

---

## Part B — OD / AL / CL / Inference Setup

### Option 1 — Docker (recommended)

#### Prerequisites

| OS | What to install |
|---|---|
| **Windows** | [Docker Desktop](https://www.docker.com/products/docker-desktop/) — enable WSL2 backend + GPU in Settings → Resources |
| **Linux** | Docker Engine + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| **Mac** | Docker Desktop (CPU only — no GPU training) |

#### Setup

```bat
:: 1. Build the image (once — ~5 GB download, ~10 min first time)
scripts\build_docker.bat

:: 2. Verify GPU is visible inside the container
docker run --rm --gpus all smartsdg info
::   SmartSDG image ready
::   torch      : 2.3.0
::   CUDA       : True (12.1)
::   GPU        : NVIDIA GeForce RTX 2080 ...

:: 3. Create pipeline_config.json (git-ignored, per-machine)
```

`pipeline_config.json` at the project root:

```json
{
  "docker_enabled": true,
  "val_real": "C:/path/to/your/val_real"
}
```

#### Start the dashboard

```bat
:: Option A — via compose (opens browser automatically)
docker compose -f docker/docker-compose.yml up dashboard

:: Option B — direct (same thing)
docker run --gpus all --shm-size=2g -p 8765:8765 \
  -v "%CD%:/workspace" smartsdg dashboard
```

Open **http://localhost:8765** — all stages (train / AL / CL) can be triggered
from the UI with Docker enabled.

---

### Option 2 — Conda environment (no Docker)

```bash
# 1. Create env
conda create -n yolov12 python=3.10
conda activate yolov12

# 2. Install PyTorch (CUDA 12.1 build)
pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining deps
pip install -r requirements.txt

# 4. Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from aas_core3 import jsonization; print('AAS ok')"
python -c "import ultralytics; print(ultralytics.__version__)"
```

> **Package names that differ from PyPI defaults:**
> - Use `aas-core3.0` (not `aas-core3.0rc02`) — wrong name installs under `aas_core3_rc02`
> - Use `ultralytics==8.3.157` to match the Docker image

Configure machine paths:

```bat
:: Check what got resolved
python config_paths.py
```

```
  isaac_python   = ...\omniverse\python.bat              [OK]
  train_python   = ...\envs\yolov12\python.exe           [OK]
  val_real       = ...\val_real                          [OK | MISSING]
```

If anything shows `MISSING`, create `pipeline_config.json`:

```json
{
  "isaac_python": "C:/path/to/omniverse/python.bat",
  "train_python": "C:/path/to/envs/yolov12/python.exe",
  "val_real":     "C:/path/to/val_real"
}
```

Then start the dashboard:

```bat
scripts\launch_dashboard.bat
```

---

## Verify the full chain

```bat
scripts\run_tests.bat        :: static checks ~10 s
scripts\run_tests.bat full   :: + real SDG + 1-epoch training ~2-5 min
```

All checks must pass before running experiments.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA: False` inside container | Enable GPU in Docker Desktop → Settings → Resources → WSL2; restart Docker |
| `OSError: No space left on device` (shm) | Add `--shm-size=2g` to the docker run command (dashboard does this automatically) |
| `Can't get attribute 'C3k2'` | Ultralytics version mismatch — rebuild image (`scripts\build_docker.bat`) |
| `ModuleNotFoundError: aas_core3` | Wrong package: use `aas-core3.0`, not `aas-core3.0rc02` |
| Out of GPU memory | Reduce `batch_size` to 4–8 in the dashboard training form (8 GB VRAM → max 8) |
| Isaac Sim slow first run | Normal — compiling shaders. Use cache on subsequent runs. |
| `docker: image not found` | Build the image first: `scripts\build_docker.bat` |
| Dashboard not opening | Check port 8765 is free: `Get-NetTCPConnection -LocalPort 8765` |
