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

## Docker for the Python stages (optional)

`docker/Dockerfile` builds an image with torch (CUDA), ultralytics and
aas_core3. Datasets, weights and configs are mounted as volumes, so the
container is stateless.

```bash
# build
docker build -t smartsdg-train -f docker/Dockerfile .

# train (Linux host with nvidia-container-toolkit)
docker run --gpus all -v "$PWD:/workspace" -w /workspace/OD_training \
    smartsdg-train python main.py --mode train \
    --aas_file ../AAS/datasets/<date>/<config>.json
```

This does **not** include Isaac Sim — generate datasets natively first, then
train/AL/infer in the container (or just keep using the conda env; the container
is only for moving to a clean GPU box).

---

## Recommended setup on a fresh machine

1. Install **Isaac Sim 4.5** (native) — only needed for SDG.
2. Create the Python env: `pip install -r requirements.txt` (or build the Docker
   image).
3. `python config_paths.py` — confirm all three paths resolve `[OK]`; if any are
   `MISSING`, set the env var or edit `pipeline_config.json`.
4. `scripts\run_tests.bat` — the self-test confirms the whole chain.
5. `scripts\launch_dashboard.bat`.
