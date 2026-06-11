# Installation Guide

This guide covers two environments:
- **Environment A** â€” SDG (Synthetic Data Generation) runs inside a Docker container on a Linux machine with an NVIDIA GPU
- **Environment B** â€” OD / AL / CL / Inference run as plain Python on any machine with a GPU (Linux or Windows)

---

## Prerequisites

### Hardware
- **NVIDIA RTX GPU** (RTX 20-series or later) â€” RTX 2080 8GB confirmed working
- 16 GB RAM minimum
- 50 GB free disk space for datasets and model checkpoints

### Software
| Component | Software | Installed Version |
|-----------|----------|-------------------|
| SDG | NVIDIA Isaac Sim 4.5 (Windows native) | 4.5.0-rc.36 |
| SDG | NVIDIA Driver | 576.80+ |
| OD / AL / CL | Python | 3.10 |
| OD / AL / CL | PyTorch | 2.3+ |
| OD / AL / CL | Ultralytics (YOLOv8/12) | 8.2+ |

> **No Docker required.** Isaac Sim is installed natively at `C:\Users\hota02\Documents\omniverse`

---

## Part A â€” SDG Setup (Isaac Sim â€” Windows Native)

Isaac Sim is **already installed** at `C:\Users\hota02\Documents\omniverse`.  
Python is at `C:\Users\hota02\Documents\omniverse\kit\python\python.exe` (invoked via `python.bat`).

### Step 1 â€” Verify Isaac Sim

```bat
C:\Users\hota02\Documents\omniverse\python.bat --version
:: Should print: Python 3.10.x
```

### Step 2 â€” Verify GPU

```bat
nvidia-smi
:: Should show: RTX 2080 Max-Q, CUDA 12.x
```

### Step 3 â€” Run a quick SDG test (5 images)

```bat
cd C:\Users\hota02\Documents\SDG_v2\SmartSDG

C:\Users\hota02\Documents\omniverse\python.bat main.py ^
  --config "src\config\config.yaml" ^
  --component_class GlueGun ^
  --light_mode Medium ^
  --num_samples 5 ^
  --load_env plane ^
  --width 1280 ^
  --height 800 ^
  --convert_to_yolo
```

> `--width` / `--height` set the render *and* YOLO label resolution for the whole run (default from `config.yaml` is 128Ã—80 â€” too small for serious training; use 1280Ã—800).  
> Use `--multi_object` instead of `--component_class` to place **all** components in every scene.

> First run takes several minutes â€” Isaac Sim compiles shaders. Subsequent runs are much faster.

Output appears at: `Datasets\yolo_experiments\SINGLE_GlueGun_Medium_plane\`

### Using the launcher script (easier)

```bat
:: Interactive â€” double-click or run with no arguments, answer the prompts:
phd_pipeline\scripts\run_sdg.bat

:: Or fully scripted: mode component light samples env width height
phd_pipeline\scripts\run_sdg.bat single GlueGun Medium 100 plane 1280 800
phd_pipeline\scripts\run_sdg.bat multi - Medium 100 plane 1280 800
```

### Light mode values

| Flag value | Intensity range |
|-----------|----------------|
| `Low` | 500 â€“ 5 000 lux |
| `Medium` | 5 000 â€“ 20 000 lux |
| `High` | 20 000 â€“ 80 000 lux |

---

## Part B â€” OD / AL / CL / Inference Setup (Python)

### Step 1 â€” Create Python environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3.10 -m venv .venv
source .venv/bin/activate
```

### Step 2 â€” Install dependencies

```bash
# Core (OD training, Active Learning, Continual Learning)
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics==8.2.2
pip install numpy==1.26.4 pandas==2.2.2 PyYAML==6.0.1 opencv-python
pip install aas-core3.0rc02

# Verify
python -c "import ultralytics; print(ultralytics.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 3 â€” Configure paths

Edit `phd_pipeline/configs/paths.env` with your local paths:

```env
SDG_ROOT=C:\Users\hota02\Documents\SDG_v2
CLUSTER_ROOT=C:\Users\hota02\Documents\sd_light_od\cluster_experiments
DATASET_ROOT=C:\Users\hota02\Documents\SDG_v2\SmartSDG\Datasets
```

### Step 4 â€” Verify training works

```bash
cd SDG_v2/OD_training
python main.py --mode train --aas_file "AAS/CA_Modul-OD_pipeline_v0.5.json"
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `nvidia-smi` not found in container | Re-run `docker` command with `--runtime=nvidia --gpus all` |
| `torch.cuda.is_available()` returns `False` | Check CUDA version matches PyTorch build |
| Isaac Sim takes a long time on first run | Normal â€” it is compiling shaders. Subsequent runs use cache. |
| `ModuleNotFoundError: ultralytics` | Activate your venv first, then `pip install ultralytics` |
| `aas_core3` import error | `pip install aas-core3.0rc02` |
| Out of GPU memory during training | Reduce `batch` size in AAS config or use `--device cpu` for testing |

