asd# CLAUDE.md — project memory for SmartSDG Pipeline v2

Read this first. It captures everything needed to work here WITHOUT re-exploring
the codebase, so keep edits surgical and avoid re-reading large files. Verify with
the self-test, not broad re-reads.

## What this is
End-to-end synthetic-data → detection pipeline, dashboard-driven, AAS-recorded.
Stages: ① SDG (Isaac Sim) → ② OD Training (YOLOv12) → ③ Inference / harvest →
④ Active Learning → ⑤ Continual Learning. One local dashboard orchestrates all,
and every run is recorded into an IEC 63278 AAS "passport" per dataset.

## Quick commands (this machine)
```bat
:: dashboard (http://127.0.0.1:8765)
scripts\launch_dashboard.bat

:: self-test — ALWAYS use to verify instead of manual exploration
scripts\run_tests.bat            :: static tier (~20 checks, ~20s) — must stay green
scripts\run_tests.bat full       :: + real SDG + train + AL (needs Isaac + GPU, ~5-10 min)
:: machine-independent subset (what CI runs; no Isaac/GPU/torch/assets needed):
"%YOLOV12_PY%" -X utf8 scripts\test_pipeline.py --ci

:: SDG (Isaac) — components/light/env via flags; config from AAS if --aas_file given
omniverse\python.bat main.py --multi_object --components Cabin,GlueGun --light_mode Medium ^
  --num_samples 100 --load_env plane --width 640 --height 480 ^
  --convert_to_yolo --drop_truncated --filter_overlaps     :: (run inside SmartSDG/)

:: training / AL / CL from an AAS passport
python OD_training/main.py --mode train --aas_file AAS\datasets\<date>\CA_Modul-OD_dataset_<name>.json
python active_learning/run_al_from_aas.py --aas_file <passport.json>
python continual_learning/main_cl.py --current_classes ... --new_classes ... --new_data_dir <dir> ...
```
`%YOLOV12_PY%` = `C:\Users\hota02\AppData\Local\anaconda3\envs\yolov12\python.exe`.

## Environments / paths
- Training/dashboard Python: conda env `yolov12` (`...\anaconda3\envs\yolov12\python.exe`) — has torch + ultralytics + aas_core3 + opencv + markdown.
- Isaac Sim Python (SDG only): `C:\Users\hota02\Documents\omniverse\python.bat`.
- Auto-resolved by `config_paths.py` (env var > `pipeline_config.json` > auto-detect > fallback). **Never hardcode paths.**
- Dashboard server: `dashboard/server.py`, stdlib-only, single job at a time, serves port 8765.
- Restart server (PowerShell): kill the process owning `Get-NetTCPConnection -LocalPort 8765`, then `Start-Process` the server hidden. **Dashboard (`server.py` / `index.html`) changes need a restart + browser hard-reload to take effect.**

## File map (where things live)
- `aas_passport.py` (repo root, stdlib) — AAS layout + `Passport` class + `record_run`.
- `config_paths.py` (repo root) — portable machine-path resolution.
- `SmartSDG/main.py` + `src/` — ① generation (`generation.py`), YOLO conversion + filters (`yolo_converter.py`), config load/merge (`config_process.py` incl. `aas_sdg_overrides`/`sdg_config_to_aas_nodes`), schema (`config_loader.py`), master config (`src/config/config.yaml`).
- `OD_training/main.py` + `src/train_eval/trainer_aas.py` — ② training; `src/aas_od_config_converter.py` turns an AAS into the YOLO training yaml.
- `active_learning/` — ④ (`run_al_from_aas.py`, `main_yolo.py`, `utils_yolo.py`, `harvest_from_inference.py`).
- `continual_learning/main_cl.py` + `cl_src/` — ⑤.
- `dashboard/server.py` + `index.html` — orchestrator + UI; live inference `/stream`, folder inference `/api/inference_images`.
- `scripts/test_pipeline.py` — the self-test (static / `--full` / `--ci` tiers). `.github/workflows/ci.yml` runs `--ci` on push/PR to main.

## Config layers (don't merge these — by design)
- `pipeline.yaml` = cross-stage DEFAULTS the dashboard pre-fills (served live via `/api/defaults`).
- `SmartSDG/src/config/config.yaml` = SDG params + `COMPONENT_USD_MAP` (the class registry; **append-only**, global IDs never reorder; Isaac reads it directly). It is the fallback source and the canonical content of the v3.0 template.
- **`AAS/CA_Modul-OD_pipeline_v3.0.json` = default template** and a **faithful 1:1 source** for SDG (generated from config.yaml, not hand-edited). SDG reads its config from an AAS via `SmartSDG/main.py --aas_file <aas>` → `config_process.aas_sdg_overrides()` (covers ALL SDG sections incl. registry + lights), config.yaml fills any gaps (`_deep_merge`); the dashboard passes the SDG tab's picked AAS (`fClassAas`, which also becomes the cloned passport's template). Regenerate v3.0 faithfully with `config_process.sdg_config_to_aas_nodes(config.yaml)` → rebuild the `OD_Module/SmartSDG` collection. Fidelity is regression-tested (`aas_sdg_overrides(v3.0) == config.yaml`). v2.0 is legacy (kept on disk; its SDG sections drifted — don't use as a config source).
- `AAS/datasets/<date>/CA_Modul-OD_dataset_<name>.json` = per-dataset record ("passport": per-stage config + measurements + append-only RunHistory), git-committed.
- `AAS/pipeline_history.json` = factory event log (git-ignored, rolling).
- `pipeline_config.json` = per-machine paths + docker flags (git-ignored).

## The AAS passport (single source of truth: `aas_passport.py`)
- `Passport.for_dataset(name)` opens (or clones-from-template, stripping provenance) a dataset's passport; `read_config(stage)` / `write_config` / `write_measurements` / `append_run` / `save` (BOM-free) let every stage read+write its slice of the SAME file. The dashboard imports the `_aas_*` helpers from here — don't redefine them.
- **CLI runs self-record**: each stage main (`SmartSDG/main.py`, `OD_training/main.py`, `active_learning/run_al_from_aas.py`, `continual_learning/main_cl.py`) calls `aas_passport.record_run(...)` (non-raising) at the tail of a successful run. The dashboard sets `SMARTSDG_SKIP_PASSPORT=1` on the stage subprocess so its own `record_to_aas` stays the single writer (no double entries). So: **dashboard run → `record_to_aas`; bare CLI run → `record_run`.**
- Dashboard Datasets tab → "AAS passport" shows the per-stage hyperparameter survey for a dataset.

## Hard invariants (do not break)
- **AAS files must be BOM-free** (Isaac yaml + aas_core3 reject BOMs). Write UTF-8 no-BOM. PowerShell `Set-Content -Encoding utf8` adds a BOM — use `[IO.File]::WriteAllText(p, s, (New-Object Text.UTF8Encoding $false))` or the Write tool.
- **Never keep an image with an unlabeled object** — occluded/truncated/degenerate objects discard the WHOLE frame (yolo_converter policies default "frame").
- **Class registry append-only** — new components append to COMPONENT_USD_MAP at the end (next free ID); existing IDs never shift.
- **Record mode**: run params + outcomes go into the per-dataset AAS; every completion also appends to `pipeline_history.json`. Keep both.
- **Templates at the AAS root are read-only SCHEMAS, never records** — cloning a passport strips provenance so template run residue is never inherited.
- Validated SDG defaults: 640×480, min_visibility 0.95, spawn 0.2, separation 0.05, IoU 0.7, containment 0.75, drop_truncated, light Medium, env plane.
- **Geometric collapse** — `filter_overlapping_boxes` drops a frame when any pair of label boxes collapses: IoU ≥ iou_threshold (boxes coincide) OR containment ≥ containment_threshold (intersection ÷ smaller-box area — a small box nested in a big one, which IoU misses). Depth-agnostic. Wired through config.yaml `YOLO_FILTERING.CONTAINMENT_THRESHOLD`, `--containment_threshold`, pipeline.yaml, and the dashboard.
- `run_yolo_conversion` returns a **2-tuple** `(out_dir, stats)`; on no rendered frames it returns `(out_dir, None)` (don't make the caller's unpack crash).
- Default model is **YOLOv12** (yolov12l.pt) everywhere user-facing.

## Stage specifics
- **SDG**: generation params come from `config.yaml`, or from an AAS when `--aas_file` is passed (config.yaml fallback). Components/light/env/samples/width/height are CLI flags. Output is always written under `SmartSDG/Datasets/...`; recorded width/height land in the passport's `GenerationParameters`.
- **Training**: reads the chosen AAS → `generate_config` → YOLO yaml. Output: `OD_training/results/yolo_run/exp{N}/<dataset>/<camera>/weights/best.pt` (env→exp: noenv/withoutenv=0, plane=1, withenv=2). The run subfolder is the **camera only** (the project dir already carries the dataset name) — repeating it overflowed Windows MAX_PATH (WinError 206) on long multi-class names. Stale ultralytics label `.cache` files are cleared before each run (a Docker-built cache stores `/workspace/...` paths and breaks a local run, and vice-versa).
- **AL**: strategies random | entropy | ppal_box | ppal_feat (default, embedding diversity). Per-camera (Replicator/Replicator_01) kept separate. Gain logged to `al_summary.csv`.
- **CL**: replay-based class-incremental. New-class data is global-ID→0-based remapped before `main_cl.py`'s offset. Model's trained classes derived from dataset-name in path (or `trained_classes.json` sidecar) — NOT `model.names` (always full registry).
- **Inference** (dashboard ③): live MJPEG `/stream` and folder `/api/inference_images`.
  - `imgsz` defaults to the model's **trained** imgsz (read from the checkpoint via `_resolve_imgsz`/`_model_trained_imgsz`), not ultralytics' fixed 640. UI "Image size" field overrides; blank = auto.
  - "align to SDG output (AAS)" toggle → `_sdg_target_for_model(model)` reads the model's dataset passport `GenerationParameters.width/height` (fallback 640×480) and `_crop_resize_to` centre-crops + resizes each frame so inference geometry matches training (no distortion). The model→passport link is derived from the model path (`…/<dataset>/<camera>/weights/best.pt` → passport `CA_Modul-OD_dataset_<dataset>.json`).
  - Live: `/api/stop_stream` releases the camera (token bump); single stream only (concurrent webcam open hard-crashes on Windows).
- **Live console**: `_reader` decodes raw bytes and folds tqdm carriage-return refreshes into one **live line** (committed history + `job["live"]`), so training shows a single updating progress line in the dashboard (local and Docker), not multi-line spam.

## Docker
- Opt-in (`pipeline_config.json` `docker_enabled`). `dockerize()` wraps the GPU Python stages (train/AL/CL) in `docker run --gpus all`, mounts the project at `/workspace`, val at `/valdata`, translates Windows paths. SDG (Isaac), harvest, live inference stay native. Image: `smartsdg` (`scripts/build_docker.bat`, `docker/`).

## Conventions
- Commit only when asked; group into focused commits. End commit body with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Use a `.git\CMTMSG.txt` file + `git commit -F` for multi-line messages (avoids PowerShell quoting breakage).
- Record **every** hyperparameter change in `CHANGELOG.md` (SDG is highly sensitive to them).
- Background long jobs; verify via the self-test or a single targeted check, not broad re-reads.
- There are usually unpushed commits; the user runs `git push` themselves.
- Shell is PowerShell (a Bash tool is also available). Prefer the Write tool for files that must be BOM-free.

## Docs map (point the user here, don't re-summarize)
`docs/DASHBOARD_GUIDE.md` (step-by-step server use + parameters + error tracking +
cluster CLI), `docs/GUIDE.md` (full guide + PDF), `docs/PIPELINE.md`, `docs/CONFIG.md`,
`docs/AAS_SCHEMA.md` (passport API + layout), `docs/HYPERPARAMETERS.md`,
`docs/LIGHTING_REALISM.md` (illumination + §8 camera-noise/realism post-processing, incl.
the standalone `camera_noise.py` CLI to perturb an existing dataset),
`docs/ACTIVE_LEARNING.md`, `docs/CONTINUAL_LEARNING.md`, `docs/PORTABILITY.md`, `docs/INSTALL.md`.
