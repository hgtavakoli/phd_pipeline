# Pipeline Technical Deep Dive

This document explains each stage in detail â€” what runs, what it produces, and how data flows between stages.

---

## Architecture Diagram

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                        AAS Configuration                             â”‚
â”‚          CA_Modul-OD_pipeline_v*.json  (drives all stages)           â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                            â”‚
          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
          â”‚  Stage 1: SDG â€” SmartSDG            â”‚
          â”‚  NVIDIA Isaac Sim 4.5.0 (Docker)    â”‚
          â”‚  â€¢ Randomised object pose           â”‚
          â”‚  â€¢ 3 lighting modes (S/M/L)         â”‚
          â”‚  â€¢ Dual camera (top-down + orbital) â”‚
          â”‚  OUTPUT: RGB images + 2D bboxes     â”‚
          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                            â”‚
          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
          â”‚  Stage 2: YOLO Conversion           â”‚
          â”‚  yolo_converter.py                  â”‚
          â”‚  â€¢ Normalise coordinates            â”‚
          â”‚  â€¢ IOU overlap filtering (0.5)      â”‚
          â”‚  OUTPUT: images/ + labels/ + .yaml  â”‚
          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                            â”‚
               â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
               â”‚                         â”‚
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚  Stage 3: OD Train â”‚    â”‚  Stage 4: Active       â”‚
  â”‚  YOLOv12            â”‚    â”‚  Learning (PPAL)       â”‚
  â”‚  AAS-driven         â”‚    â”‚  Phase 1: Entropy      â”‚
  â”‚  Multi-GPU DDP      â”‚    â”‚  Phase 2: CCMS+KMeans  â”‚
  â”‚  OUTPUT: best.pt    â”‚    â”‚  OUTPUT: selected imgs â”‚
  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â”‚                         â”‚
               â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                            â”‚
          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
          â”‚  Stage 5: Continual Learning        â”‚
          â”‚  Class-Incremental (Replay)         â”‚
          â”‚  â€¢ Balanced batch sampler           â”‚
          â”‚  â€¢ Replay ratio: 0.0â€“1.0            â”‚
          â”‚  OUTPUT: updated best.pt            â”‚
          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                            â”‚
          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
          â”‚  Stage 6: Inference                 â”‚
          â”‚  live_inference.py                  â”‚
          â”‚  Webcam / RTSP / HTTP               â”‚
          â”‚  OUTPUT: real-time detections       â”‚
          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## Stage 1 â€” SDG: Synthetic Data Generation

**Location:** `SDG/SmartSDG/`  
**Entry:** `main.py`  
**Requires:** NVIDIA Isaac Sim 4.5.0 Docker container

### What it does

Launches NVIDIA Omniverse Isaac Sim and uses the **Replicator** framework to render photorealistic scenes of industrial truck components.

### Key parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--component_class` | required | Object to render (e.g. `GlueGun`, `Cabin`) |
| `--light_mode` | required | `S`=5 000Â±500, `M`=20 000Â±5 000, `L`=80 000Â±20 000 lux |
| `--num_samples` | `10` | Number of images to generate |
| `--load_env` | `false` | `plane` / `true` (3D scene) / `false` (void) |
| `--multi_object` | off | Scatter all 8 classes randomly in one scene |
| `--seed` | random | Fix for reproducibility |
| `--convert_to_yolo` | off | Auto-run YOLO converter after generation |

### Cameras

- **Camera 1** â€” Fixed top-down view, consistent framing
- **Camera 2** â€” Orbital view, randomised position each frame (azimuth Â± 45Â°, elevation 15â€“90Â°)

### Output structure

```
Datasets/unified_render_output/SINGLE_GlueGun_M_plane/
    Replicator/
        camera1_rgb_0000.png
        camera1_bbox2d_0000.npy
        camera2_rgb_0000.png
        camera2_bbox2d_0000.npy
        ...
    metadata.txt
```

---

## Stage 2 â€” YOLO Conversion

**Location:** `SDG/SmartSDG/src/yolo_converter.py`

### What it does

Reads Isaac Sim's numpy bounding box arrays and converts to YOLO label format:
`<class_id> <cx> <cy> <w> <h>` (all normalised 0â€“1)

Optionally removes images where any two boxes overlap with IoU > 0.5.

### Output structure

```
Datasets/yolo_experiments/SINGLE_GlueGun_M_plane/
    images/train/  images/val/  images/test/
    labels/train/  labels/val/  labels/test/
    dataset.yaml
```

---

## Stage 3 â€” Object Detection Training

**Location:** `SDG/OD_training/`  
**Entry:** `main.py` (AAS-driven) or `cluster_experiments/train_yolov12.py` (cluster)

### What it does

Trains YOLOv12 models using Ultralytics. Supports:
- Multiple datasets in one run (iterates experiment configs from AAS)
- Multi-camera experiments (camera1, camera2, both)
- Periodic evaluation every N epochs â†’ logs to CSV
- DDP multi-GPU via `torchrun` (2+ GPUs)

### AAS config keys used

```json
"ObjectDetection": {
  "model_size": "n",          // nano/small/medium/large/xl
  "epochs": 100,
  "batch": 16,
  "imgsz": 640,
  "dataset_paths": [...],
  "experiment_name": "exp1"
}
```

### Output

```
OD_training/results/experiment_outputs/exp1/
    SINGLE_GlueGun_Medium_plane/
        weights/best.pt
        weights/last.pt
        evaluation_summaries/
            eval_summary_on_test_(best).txt
```

---

## Stage 4 â€” Active Learning (PPAL)

**Location:** `SDG/active_learning/`  
**Entry:** `main_yolo.py`  
**Algorithm:** Plug-and-Play Active Learning (CVPR 2024)

### What it does

Selects the most informative unlabelled images to annotate next, minimising annotation cost.

### Two-phase algorithm

**Phase 1 â€” Uncertainty Sampling**
1. Run current model on unlabelled pool
2. Compute Shannon entropy of detection confidences per image
3. Apply class-specific difficulty coefficients
4. Score = entropy Ã— difficulty

**Phase 2 â€” Diversity via CCMS**
1. Category-Conditioned Matching Similarity: compares images by shared classes + bbox geometry
2. K-Means++ clustering on similarity scores
3. Select one representative from each cluster â†’ balanced, diverse batch

### Workflow

```
Round 0: train on 10% seed data
Round 1: select 10 most uncertain images â†’ annotate â†’ retrain
Round 2: select next 10 â†’ annotate â†’ retrain
...repeat for N rounds
```

### Output

```
active_learning/workspace/
    round_0/  round_1/  round_2/ ...
        images/train/  labels/train/
    selected_for_annotation/
        round_1_selected.txt
```

---

## Stage 5 â€” Continual Learning

**Location:** `SDG/continual_learning/`  
**Entry:** `run_from_aas.py`

### What it does

Extends an existing model to detect new object classes without forgetting previously learned ones (catastrophic forgetting problem).

### Strategy: Replay-based Class-Incremental Learning

1. Load existing model trained on classes A, B, C
2. Collect new data for class D
3. **Balanced Batch Sampler**: each batch contains both new data (class D) + replay samples (classes A, B, C)
4. Replay ratio is configurable (0.0 = no replay, 1.0 = equal new/old)
5. Retrain â€” model now detects A, B, C, D

### AAS config keys used

```json
"ContinualLeaning": {
  "base_model_path": "...",
  "new_class_data": "...",
  "replay_ratio": 0.5,
  "epochs": 50
}
```

---

## Stage 6 â€” Inference

**Location:** `SDG/inference/live_inference.py`

### What it does

Loads a trained `.pt` model and runs real-time detection on a video source.

### Usage

```bash
python live_inference.py --model weights/best.pt --source 0       # webcam
python live_inference.py --model weights/best.pt --source rtsp://...  # IP camera
python live_inference.py --model weights/best.pt --source http://...  # HTTP stream
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--conf` | `0.3` | Confidence threshold |
| `--hide-labels` | off | Hide class names on boxes |
| `--hide-conf` | off | Hide confidence scores |

---

## Stage 7 â€” AAS Configuration

**Location:** `SDG/AAS/`

The **Asset Administration Shell** (AAS, IEC 63278) is a standardised digital twin format used here to store all pipeline configuration in a single, versioned JSON file.

### Tree structure

```
CA_Modul
â””â”€â”€ OD_Module
    â”œâ”€â”€ ObjectDetection   â† Training hyperparameters, paths, experiments
    â”œâ”€â”€ ActiveLearning    â† AL rounds, pool paths, selection count
    â””â”€â”€ ContinualLeaning  â† Base model, new class data, replay ratio
```

### Why AAS?

- Single source of truth for the entire pipeline
- Version-controlled experiment configs
- Reproducibility: anyone running `v0.5.json` gets the same experiment
- Compatible with the AASX Package Explorer (included in `AAS/`)

---

## Data Flow Summary

| Input | Stage | Output |
|-------|-------|--------|
| 3D USD assets | SDG | Raw RGB + bbox numpy arrays |
| Raw Isaac output | YOLO Convert | YOLO images/ labels/ dataset.yaml |
| YOLO dataset | OD Train | best.pt, CSV metrics |
| best.pt + unlabelled pool | Active Learning | Selected image list for annotation |
| Annotated images | OD Retrain | Improved best.pt |
| best.pt + new class data | Continual Learning | Extended best.pt |
| Extended best.pt | Inference | Real-time bounding boxes |

