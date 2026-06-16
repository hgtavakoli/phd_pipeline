# Running on Another Machine — Portability & Docker

## TL;DR

The pipeline no longer hard-codes any machine path. On a new machine you either:

1. **Set three environment variables**, or
2. **Copy `pipeline_config.example.json` → `pipeline_config.json`** and fill in
   the paths, or
3. **Do nothing** — auto-detection finds standard install locations.

Check what got resolved at any time:

```bat
python config_paths.py
```
```
  isaac_python   = ...\omniverse\python.bat              [OK]
  train_python   = ...\envs\yolov12\python.exe           [OK]
  val_real       = ...\val_real                          [OK | MISSING]
```

The three paths:

| Key | What | Env var |
|---|---|---|
| `isaac_python` | Isaac Sim's `python.bat` (SDG only) | `SMARTSDG_ISAAC_PYTHON` |
| `train_python` | Python with torch+ultralytics+aas_core3 (OD/AL/CL/inference) | `SMARTSDG_TRAIN_PYTHON` |
| `val_real` | folder of real validation images (optional) | `SMARTSDG_VAL_REAL` |

Resolution order per path: **env var → `pipeline_config.json` → auto-detect →
fallback**. `pipeline_config.json` is git-ignored so each machine keeps its own.

---

## Can Isaac Sim 4.5 run in Docker? — honest answer

**Partly, and not on this Windows setup.** Here is the real situation so you can
decide:

- NVIDIA *does* publish an Isaac Sim **Docker image**, but it is **Linux-only**,
  ~20 GB, and requires the **NVIDIA Container Toolkit** with GPU passthrough.
- Your current SDG runs on **Windows-native Isaac Sim** (the way NVIDIA ships it
  for Windows). Windows Isaac Sim does **not** containerize; the Docker image
  would only run under **WSL2 + GPU passthrough** (Linux inside Windows), which
  is a large, fragile setup and would not reuse your existing working install.
- So: **Dockerizing the SDG stage on Windows is not worth it.** The native
  install already works, and the path-config change above already makes it
  portable to any Windows machine with Isaac Sim installed.

**What *is* worth containerizing: the Python stages.** OD training, Active
Learning, Continual Learning and inference are plain PyTorch/Ultralytics — they
containerize cleanly and run on any machine (Linux or Windows+WSL2) with an
NVIDIA GPU. A `Dockerfile` for that is provided (see below). The split is:

```
  SDG (Isaac Sim)            →  native install (Windows or Linux), path-configured
  OD / AL / CL / Inference   →  Docker (portable) OR the same native conda env
```

---

## Docker for the Python stages (recommended for new machines)

`docker/Dockerfile` builds a self-contained image covering **OD Training,
Active Learning, Continual Learning, Inference and the Dashboard**. Datasets,
weights and configs are bind-mounted at runtime — the image itself stays small
and stateless. Isaac Sim is the only stage that stays native.

### What you need on the host

| Requirement | Windows | Linux |
|---|---|---|
| Docker Desktop | ✅ | Docker Engine |
| GPU passthrough | Docker Desktop → Settings → Resources → WSL2 GPU | NVIDIA Container Toolkit |
| NVIDIA driver | 525+ | 525+ |

### One-time build

```bat
:: Windows
scripts\build_docker.bat

:: Linux / Mac
docker build -t smartsdg -f docker/Dockerfile .
```

First build downloads ~5 GB (PyTorch CUDA base). Subsequent builds use the
layer cache and take ~30 s.

### Start the dashboard

```bash
docker compose -f docker/docker-compose.yml up dashboard
```

Then open **http://localhost:8765**. All pipeline stages can be triggered from
the dashboard UI with Docker enabled.

### Enable Docker mode in the dashboard

In the dashboard **⚙ Execution** section:

1. Tick **"run training/AL/CL in Docker"** — the toggle is saved to
   `pipeline_config.json` (`docker_enabled: true`).
2. Training, AL, and CL runs are then automatically wrapped in
   `docker run --gpus all --shm-size=2g`, with project files mounted at
   `/workspace` and val-real images at `/valdata`. Path translation from
   Windows host paths to container paths is handled automatically.

Or set it manually:

```json
// pipeline_config.json  (git-ignored, per-machine)
{
  "docker_enabled": true,
  "val_real": "C:/path/to/your/val_real"
}
```

### One-off stage commands (without dashboard)

```bash
# OD Training
docker run --gpus all --shm-size=2g \
  -v "$PWD:/workspace" \
  -e SMARTSDG_WIN_ROOT="$PWD" \
  smartsdg train --aas_file AAS/datasets/<date>/<config>.json

# Active Learning
docker run --gpus all --shm-size=2g -v "$PWD:/workspace" smartsdg al

# Continual Learning
docker run --gpus all --shm-size=2g -v "$PWD:/workspace" smartsdg cl

# Debug shell inside the container
docker compose -f docker/docker-compose.yml run --rm shell
```

### Entrypoint modes

| Command | What runs |
|---|---|
| `smartsdg info` | Print torch/CUDA/GPU versions (default) |
| `smartsdg dashboard` | Start the dashboard on :8765 |
| `smartsdg train [args]` | `OD_training/main.py --mode train [args]` |
| `smartsdg al [args]` | `AL/main_al.py [args]` |
| `smartsdg cl [args]` | `CL/main_cl.py [args]` |
| `smartsdg inference [args]` | `inference/run_inference.py [args]` |
| `smartsdg shell` | Interactive bash |
| `smartsdg python script.py` | Run any python script |

### Known limitations

- **Live webcam inference** does not work in Docker on Windows — USB passthrough
  is unreliable. Use native inference for the live camera tab.
- **Batch size**: YOLOv12l on 8 GB VRAM needs `batch_size ≤ 8`. The template
  AAS defaults to 8; adjust in the dashboard form for larger GPUs.
- **Shared memory**: `--shm-size=2g` is required for PyTorch's multi-worker
  dataloader. The dashboard injects this automatically.

---

## Recommended setup on a fresh machine

### With Docker (recommended — no Python install needed)

1. Install **Isaac Sim 4.5** natively (SDG stage only).
2. Install **Docker Desktop** + enable GPU (WSL2 on Windows, nvidia-container-toolkit on Linux).
3. `git clone <repo> && cd SDG_v2`
4. `scripts\build_docker.bat` (Windows) or `docker build -t smartsdg -f docker/Dockerfile .`
5. Create `pipeline_config.json` with `docker_enabled: true` and your `val_real` path.
6. `docker compose -f docker/docker-compose.yml up dashboard` → open http://localhost:8765

### Without Docker (conda env)

1. Install **Isaac Sim 4.5** natively.
2. `conda create -n yolov12 python=3.10 && conda activate yolov12`
3. `pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121`
4. `pip install -r requirements.txt`
5. `python config_paths.py` — confirm all three paths resolve `[OK]`.
6. `scripts\run_tests.bat` — self-test confirms the whole chain.
7. `scripts\launch_dashboard.bat`.
