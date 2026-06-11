# SmartSDG Pipeline — v2 (Working Copy)

> **Author:** Hooman Tavakoli Ghinani
> **Version:** 2.1.0 — see [CHANGELOG.md](CHANGELOG.md)
> **Frozen reference:** the original, validated v1 lives untouched at `C:\Users\hota02\Documents\SDG\` — never modify it.

End-to-end research pipeline: **Synthetic Data Generation (NVIDIA Isaac Sim) → YOLO dataset → Object Detection training → Active Learning → Continual Learning → Live Inference**, all driven by AAS (Asset Administration Shell, IEC 63278) configurations.

```
[Isaac Sim] → [YOLO Dataset] → [Train YOLOv12] → [Active Learning] → [Continual Learning] → [Inference]
     SDG          convert            OD                 AL                   CL                Deploy
                          AAS configs drive every stage
```

---

## Repository layout

```
SDG_v2/
├── README.md                  ← you are here
├── CHANGELOG.md               ← what changed vs. v1 (keep it updated!)
├── VERSION
├── requirements.txt           ← Python deps for OD / AL / CL / inference
├── .gitignore
│
├── docs/
│   ├── INSTALL.md             ← environment setup, step by step
│   ├── PIPELINE.md            ← technical walkthrough of all stages
│   ├── HYPERPARAMETERS.md     ← every SDG knob explained (read before changing!)
│   └── LEGACY_README_SmartSDG.md
│
├── scripts/
│   └── run_sdg.bat            ← interactive SDG launcher (validated defaults)
│
├── SmartSDG/                  ← STAGE 1-2: generation + YOLO conversion
│   ├── main.py                ← entry point (all CLI flags documented in --help)
│   ├── src/
│   │   ├── generation.py      ← Replicator graph: spawning, lighting, cameras
│   │   ├── yolo_converter.py  ← npy→YOLO conversion + quality filters
│   │   ├── config_process.py  ← YAML → AppConfig loader
│   │   ├── config_loader.py   ← dataclass schema of the config
│   │   └── config/config.yaml ← MASTER CONFIG (validated hyperparameters)
│   ├── assets/                ← USD models (environment + 9 components)
│   ├── Datasets/              ← generated output (git-ignored)
│   └── legacy/                ← dead code quarantined during v2 cleanup (safe to delete)
│
├── OD_training/               ← STAGE 3: YOLOv8/v12 training (AAS-driven)
├── active_learning/           ← STAGE 4: PPAL active learning (CVPR 2024)
├── continual_learning/        ← STAGE 5: replay-based class-incremental learning
├── inference/                 ← STAGE 6: live webcam/IP-camera detection
├── AAS/                       ← STAGE 7: AAS JSON configs (versioned v0.1…v0.8)
└── paper_visualisation/       ← figures and plots for publications
```

---

## Quick start

### Generate a dataset (Isaac Sim required)

```bat
:: Interactive — answers prompts, uses validated defaults:
scripts\run_sdg.bat

:: Or explicit:
cd SmartSDG
C:\Users\hota02\Documents\omniverse\python.bat main.py ^
  --config "src\config\config.yaml" ^
  --multi_object --components Cabin,GlueGun,BLid ^
  --light_mode Medium --num_samples 100 --load_env plane ^
  --width 640 --height 480 ^
  --convert_to_yolo --filter_overlaps --drop_truncated
```

Every run ends with a **FINAL DATASET SUMMARY** telling you exactly how many
frames were kept and why each discarded frame was dropped.

### Train / AL / CL / inference

```bash
cd OD_training        && python main.py --mode train --aas_file "AAS/CA_Modul-OD_pipeline_v0.5.json"
cd active_learning    && python main_yolo.py
cd continual_learning && python run_from_aas.py "AAS/CA_Modul-OD_pipeline_v0.7.json"
cd inference          && python live_inference.py --model weights/best.pt --source 0 --conf 0.3
```

---

## The 9 object classes

| ID | Class | | ID | Class | | ID | Class |
|----|-------|-|----|-------|-|----|-------|
| 0 | Ylid | | 3 | GlueGun | | 6 | FChassis |
| 1 | YTrailer | | 4 | BLid | | 7 | Rchassis |
| 2 | Cabin | | 5 | BTrailer | | 8 | BlackCabin |

Class IDs are **global and fixed** (derived from `COMPONENT_USD_MAP` order in
`config.yaml`). Subset datasets keep the same IDs — never renumber.

---

## Data-quality guarantees (v2 invariants)

These rules are enforced in `yolo_converter.py` — keep them when modifying:

1. **Never drop only a bounding box.** If an object's label must be removed
   (occluded, border-cut, degenerate), the **whole frame is discarded**.
   An image containing a visible-but-unlabeled object poisons OD training.
2. Every kept image contains **only fully-visible (≥95%), fully-labeled,
   border-free objects** under the default filter settings.
3. Class IDs are global across all datasets (see table above).

## Validated hyperparameter defaults

Empirically validated on 2026-06-10 (see `docs/HYPERPARAMETERS.md` for what
each one does and *why* these values):

| Parameter | Value | Where |
|---|---|---|
| Resolution | 640 × 480 | CLI `--width/--height` |
| `min_visibility` | 0.95 | `main.py` default |
| `truncation_policy` / `occlusion_policy` | frame | `main.py` default |
| `iou_threshold` (overlap filter) | 0.7 | `config.yaml` |
| `multi_spawn_radius` | 0.2 m | `config.yaml` |
| `min_separation` | 0.05 m | `main.py` default |

> ⚠ **SDG is highly sensitive to these.** Change one at a time, regenerate a
> small batch (e.g. 20 samples), inspect images + FINAL DATASET SUMMARY, and
> record the change in [CHANGELOG.md](CHANGELOG.md) before scaling up.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/INSTALL.md](docs/INSTALL.md) | Isaac Sim + Python environment setup |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Deep technical walkthrough of all 7 stages |
| [docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md) | Every SDG parameter: meaning, effect, validated value |

## Requirements

- **SDG**: NVIDIA Isaac Sim 4.5 (Windows native, `C:\Users\hota02\Documents\omniverse`), RTX GPU
- **OD/AL/CL**: Python 3.10, `pip install -r requirements.txt`

## Git / what is NOT in the repository

The repo contains **code, configs, and docs only**. Intentionally excluded by
`.gitignore` (too large / reproducible):

| Excluded | Where to get it |
|---|---|
| `SmartSDG/assets/` (1.1 GB USD models) | copy from this machine: `C:\Users\hota02\Documents\SDG_v2\SmartSDG\assets` (or the v1 original) |
| `*.pt` model weights | downloaded by Ultralytics on first run, or retrained |
| `SmartSDG/Datasets/`, training `results/`, AL `output/` | regenerated by the pipeline |

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2025-2026 Hooman Tavakoli Ghinani
