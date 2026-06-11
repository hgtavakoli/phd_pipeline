# SDG Hyperparameter Guide

Every knob that affects dataset quality, what it does, and the validated value.
**Read this before changing anything** — SDG output is highly sensitive, and a
single bad value silently degrades downstream OD training.

There are three groups, applied at different stages:

```
GROUP 1: scene placement  →  render  →  GROUP 2: quality filters  →  GROUP 3: renderer quality
(prevent problems)                       (discard bad frames)         (image fidelity/speed)
```

---

## Group 1 — Generation-time (how the scene is BUILT)

These prevent problems before rendering. Nothing is deleted by them.

| Parameter | Validated | Where | Effect |
|---|---|---|---|
| `--spawn_radius` | **0.2 m** | CLI / `config.yaml: multi_spawn_radius` | Max distance of objects from scene center. Smaller → objects stay inside camera view (no border cuts). Does **not** change object pixel size. |
| `--min_separation` | **0.05 m** | CLI | Minimum XY distance between object **centers** (collision-aware rejection sampling). Keeps objects from sitting in front of each other → prevents occlusion at the source. `0` disables. |
| `truck_scale` | 0.01 | `config.yaml` | World scale of components. **Never change** — 0.1 makes objects 10× too large/close. |
| `orbit_radius` | 1.5 | `config.yaml` | Camera 2 distance. Larger → smaller objects on screen (hurts OD). Prefer tuning `spawn_radius` instead. |
| `focal_length` | 48.0 | `config.yaml` | Camera lens. Larger → zoomed in (objects bigger, FOV narrower). |
| `multi_spawn_z_variation` | 0.3 | `config.yaml` | Vertical spread of spawned objects. |
| `LIGHT_INTENSITY_MAP` | Low/Medium/High | `config.yaml` | Mean/std lux per `--light_mode`. Keys MUST be `Low`/`Medium`/`High`. |

**Rule of thumb:** if too many frames are being discarded by Group 2, fix it
here (smaller spawn radius, larger separation), don't loosen the filters.

---

## Group 2 — Filter-time (what is KEPT after rendering)

Applied during YOLO conversion. **v2 invariant: a filter never removes only a
label — if any object fails, the whole frame is discarded** (an image with a
visible-but-unlabeled object poisons OD training).

| Parameter | Validated | Measures | Frame discarded when… |
|---|---|---|---|
| `--min_visibility` | **0.95** | Replicator `occlusionRatio` — real 3D occlusion **by other objects** | any object is > 5% hidden behind another |
| `--drop_truncated` | **on** | bbox touching the image border | any object is partially **outside the camera frame** |
| `--filter_overlaps` + `--iou_threshold` | **on, 0.7** | IoU of final YOLO **label rectangles** | two boxes overlap ≥ 0.7 |
| (automatic) | always | degenerate zero-area boxes | an object exists but no valid box can be written |

### min_visibility vs filter_overlaps — not the same thing!
- `min_visibility` = true 3D occlusion (is the object hidden?). The
  scientifically meaningful one.
- `filter_overlaps` = 2D rectangle intersection. Two fully visible side-by-side
  objects can still have overlapping *boxes*. Coarse safety net only.
- With `min_visibility 0.95` active, the overlap filter rarely fires.

### Policies
`--occlusion_policy` / `--truncation_policy`: `frame` (default — discard whole
image) or `box` (remove only the label — **avoid**, leaves unlabeled objects).

---

## Group 3 — Renderer / ray-casting quality (`config.yaml: SIMULATION_CONFIG`)

| Parameter | Validated | Effect |
|---|---|---|
| `renderer` | PathTracing | Physically accurate. `RayTracedLighting` = faster, less realistic. |
| `samples_per_pixel_per_frame` | 8 | Rays per pixel. Main quality/speed knob. 32–64 for final datasets, 8 for iteration (denoiser compensates). |
| `max_bounces` | 32 | Light path depth. |
| `max_specular_transmission_bounces` | 6 | Bounces through glass/specular. |
| `max_volume_bounces` | 4 | Volumetric media. |
| `denoiser` | True | OptiX AI denoiser — lets low spp look clean. |

Advanced carb settings (firefly filter, totalSpp, DLSS mode, RTSubframes…) go
in `config.yaml: RTX_CARB_SETTINGS` — applied automatically before the
Replicator graph builds. Commented examples are in the config file.

---

## Validated baseline command (2026-06-10)

```bat
C:\Users\hota02\Documents\omniverse\python.bat main.py ^
  --config "src\config\config.yaml" ^
  --multi_object --components Ylid,YTrailer,Cabin,GlueGun,BLid,BTrailer,FChassis ^
  --light_mode Medium --num_samples 100 --load_env plane ^
  --width 640 --height 480 ^
  --convert_to_yolo --filter_overlaps --drop_truncated
```

(min_visibility 0.95, policies frame, IoU 0.7, spawn 0.2, separation 0.05 are
defaults — only the switches need passing.)

## How to change a hyperparameter safely

1. Change **one** parameter.
2. Generate a small batch (`--num_samples 20`).
3. Inspect: images visually + the FINAL DATASET SUMMARY (discard rates).
4. Record old → new value and observed effect in `CHANGELOG.md`.
5. Only then scale up to full generation.
