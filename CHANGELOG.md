# Changelog

All notable changes to this pipeline are documented here.
Format: [Keep a Changelog](https://keepachangelog.com). Record **every**
hyperparameter change — SDG results are highly sensitive to them.

## [2.9.1] — 2026-06-18

### Camera-realism post-processing for an existing dataset (no re-generation)

- `SmartSDG/src/camera_noise.py` gains `apply_camera_realism()` + `process_dataset()`
  + a CLI: apply subtle, randomized real-camera perturbations (sensor noise, lens
  softness, exposure/contrast + white-balance drift, JPEG artefacts) to an
  already-generated YOLO dataset, instead of re-running SDG with realism on.
- PHOTOMETRIC only — geometry is untouched, so **labels are copied unchanged**.
  Default writes a parallel `<name>_real` dataset (clean originals preserved);
  `--in-place` overwrites. Seeded RNG for reproducibility; per-image jitter.
- Usage: `python SmartSDG/src/camera_noise.py <dataset_dir> [--out <dir>]
  [--in-place] [--gaussian 6] [--blur 0.5] [--brightness 0.12] [--contrast 0.12]
  [--color 0.05] [--jpeg 85] [--seed 0]`.
- Self-test: new `camera-realism post-processing` check.
- **Train a post-processed dataset without a new passport**, two ways:
  - `OD_training/main.py --data_dir <folder>` overrides the AAS `base_data_dir`,
    so you reuse an existing passport but train on the perturbed copy.
  - `camera_noise.py` creates a passport for the new dataset **by default** (cloned
    from the v3.0 template, `base_data_dir` → the new folder) so it trains like
    any other dataset, including headless/cluster. `--no-make-aas` to skip.

## [2.9.0] — 2026-06-18

### Training curve chart, cluster CLI, and headless-friendly paths

- **Dashboard training curve**: Results tab now has a 📈 line chart (run picker +
  Refresh) plotting per-epoch **precision / recall / mAP50 / mAP50-95** from a
  run's `results.csv`. New `/api/training_curve` endpoint + `training_curve()`
  reader; chart drawn on a plain `<canvas>` (no external library). Refresh while
  training to watch it climb.
- **Custom validation set → dataset AAS** (confirmed): setting a custom
  `val_data_path` in the Training tab writes it into that dataset's passport
  (record mode, `build_train_cmd`) — the passport already carries a
  `val_data_path` property, so the override persists with the run.
- **Cluster / headless training**: `OD_training/main.py` now anchors all I/O to
  the script location (`OD_ROOT`), not the cwd, so the same command works from
  the repo root under `srun`; outputs always land in `OD_training/results/`. The
  `--aas_file` accepts absolute or repo-/OD-/cwd-relative paths (`_resolve_aas`),
  and its default is the v3.0 template. New `scripts/train_cluster.sh` wraps a
  SLURM/container run (installs `aas-core3.0` if the image lacks it, prints GPU
  info, passes extra args through).
- **Fix**: `do_POST` declared `global ISAAC_PYTHON, TRAIN_PYTHON` after using
  them (latent `SyntaxError`); moved the declaration to the top of the handler.
- **Headless path alignment** (`OD_training/main.py` `_localize_paths`): a
  passport recorded on Windows stores absolute `C:\...` data paths that don't
  exist on a Linux cluster. Training now re-roots any repo-internal data
  (`SmartSDG/Datasets/...`) under THIS repo, so the same passport + command find
  the data wherever the repo is mounted. External real val/test are overridable
  via `SMARTSDG_VAL_REAL` / `SMARTSDG_TEST_REAL`; if the val set is still missing,
  the trainer falls back to the training set (with a warning) so the run
  completes. `trainer_aas` uses that resolved val dir.
- Self-test: new `training-curve from results.csv` + `cluster path localization`
  checks. Full static 21/21.
- **`--val_dir` / `--test_dir` CLI flags** on `OD_training/main.py` to point the
  validation / test set at a cluster folder, e.g.
  `--val_dir /netscratch/.../data/val_new_2`. Precedence: CLI flag ▸
  `SMARTSDG_VAL_REAL` env ▸ AAS `val_data_path` ▸ training set (fallback).

## [2.8.6] — 2026-06-17

### Inference — align frames to the SDG geometry from the AAS + dropdown overflow fix

- **AAS-driven frame alignment**: a new "align to SDG output (AAS)" toggle (live
  stream + folder inference) reads the model's dataset passport
  (`GenerationParameters.width/height` via `_sdg_target_for_model`) and
  centre-crops + resizes each frame to that exact geometry (`_crop_resize_to`,
  fallback 640×480) before detection — so a real camera/image matches the
  training preprocessing with no distortion. Folder inference thumbnails are built
  from the aligned frame so the normalised box overlay stays correct.
- **Dropdown overflow**: long AAS/model paths no longer push selects out of the
  card — `select` is capped at `max-width:100%` with ellipsis, and option labels
  are shortened (`optLabel`: dataset/camera or de-boilerplated name, middle-
  truncated) while the full path stays as the value + hover title.
- Self-test: new `inference frame alignment to SDG` check.

## [2.8.5] — 2026-06-17

### Dashboard — per-camera dataset preview + inference imgsz aligned to training

- **Dataset preview** can now be filtered per camera: a Replicator / Replicator_01
  radio selector (shown only for multi-camera datasets) drives `/api/preview`
  (`camera=` param → `dataset_preview` filters by camera subfolder). The Edit-
  labels callback preserves the selected camera.
- **Inference image size** now matches the model's training imgsz instead of
  ultralytics' fixed 640 default. New `_resolve_imgsz` / `_model_trained_imgsz`
  read the imgsz from the checkpoint; both live streaming (`/stream`) and folder
  inference (`/api/inference_images`) pass it to `model.predict`. An optional
  **Image size** field (blank = auto-match training) is added to both inference
  modes, plus a **match training 4:3 (640×480)** capture toggle for webcams so
  the field of view mirrors the SDG data. (Raw frame dims need not equal training
  dims — YOLO letterboxes — but imgsz should, which this now guarantees.)
- Self-test: new `inference imgsz matches training` check.

## [2.8.4] — 2026-06-17

### SDG — clean handling when generation produced no frames (no more TypeError)

- When the render output had no `rgb`/`bounding_box_2d_tight` folders,
  `run_yolo_conversion` returned a bare path on its early-return while the caller
  unpacked `dir, stats = ...` — crashing with `TypeError: cannot unpack
  non-iterable WindowsPath object` and masking the real issue.
- It now returns `(dir, None)` and prints a clear message that the **generation**
  step produced no frames (not a conversion problem), pointing at likely causes
  (component/USD load failure, 0 samples). Self-test: new `converter no-data
  returns tuple` check.

## [2.8.3] — 2026-06-17

### Training — fix "no labels in cache" when switching between Docker and local

- ultralytics label `.cache` files store ABSOLUTE image paths. A cache written
  inside Docker stores `/workspace/...` paths; reusing the same dataset folder
  for a native run then fails ("can't find images with proper labels") even
  though images+labels sit together in the dir — and vice versa.
- `trainer_aas.py` now clears stale label caches (the sibling `<dir>.cache` and
  any inside the data dir) for the train/val/test paths before each run, so
  ultralytics rescans against the real local paths. Rebuilding the cache is
  cheap.
- Self-test: new `stale label cache cleanup` check.

## [2.8.2] — 2026-06-17

### Training — fix Windows MAX_PATH (WinError 206) on long dataset names

- A run's output folder repeated the (long) dataset name twice — `project` dir =
  `<dataset>` and run `name` = `<dataset>_<camera>` — so for multi-class datasets
  the path to `weights/` blew past Windows' 260-char limit and training crashed
  (`FileNotFoundError: [WinError 206] filename or extension is too long`).
- `trainer_aas.py`: the run subfolder is now just the **camera** (the project dir
  already carries the dataset name); the periodic-eval folder dropped the name
  duplication too (`eval_epoch_<n>`, logged under the camera subdir). Example
  path drops from 256 → 175 chars. Weights/results locations are unchanged in
  spirit (`results/yolo_run/exp{N}/<dataset>/<camera>/weights/best.pt`) and the
  metric readers find them via rglob, so AAS provenance still works.
- `trainer_aas.py` added to the self-test compile check.

## [2.8.1] — 2026-06-17

### Dashboard — training progress shows as ONE updating line, not multi-line spam

- **Fix**: the live-console reader used text-mode `readline`, whose universal-
  newline handling turned every tqdm carriage-return refresh into a separate
  line — so one epoch's progress bar became dozens of lines (local and Docker,
  since both pipe through the dashboard as a non-TTY).
- `_reader` now reads raw bytes and folds output via `_ingest_output`: a `\r`
  refresh overwrites a single **live line**, a `\n` finalizes it (one committed
  line per epoch), and CRLF is handled without blank lines. `/api/logs` returns
  the `live` line; the frontend redraws it each poll on top of the committed
  history.
- Docker needs no `-t` (keeping it a pipe also avoids ANSI colour codes leaking
  into the log).
- Self-test: new `console progress single-line` check.

## [2.8.0] — 2026-06-17

### AAS v3.0 — a faithful, complete single source for the whole pipeline

- **New default template `AAS/CA_Modul-OD_pipeline_v3.0.json`**, generated
  programmatically from `config.yaml` (not hand-edited) so its SDG sections are a
  **1:1 faithful source** — fixing the v2.0 drift (USD names now in `value` not
  just `valueId`, `LIGHT_INTENSITY_MAP` order corrected, and the previously
  missing `LIGHT_TEMPERATURE_MEAN/STD` + `RTX_CARB_SETTINGS` added). `TEMPLATE_AAS`
  now points to v3.0 in the dashboard, the passport module, and the self-test;
  the SDG/training pickers default to v3.0. v2.0 stays on disk as legacy.
- **Converter now covers all SDG sections** (`config_process`): `aas_sdg_overrides`
  reads `paths`, `generation_params`, `COMPONENT_USD_MAP`, `YOLO_CONVERSION`,
  `YOLO_FILTERING`, `LIGHT_INTENSITY_MAP`, `LIGHT_TEMPERATURE_*`, `LIGHT_POSITIONS`,
  `SIMULATION_CONFIG`, `RTX_CARB_SETTINGS`. The reader handles both `valueId`
  (registry) and JSON-encoded list leaves, so it works on v2.0 and v3.0.
- **New builder** `config_process.sdg_config_to_aas_nodes()` (`config.yaml -> AAS
  nodes`) — the inverse of the reader; used to (re)generate v3.0 faithfully.
- **Fidelity guarantee, regression-tested**: `aas_sdg_overrides(v3.0) ==
  config.yaml` for every section, and the builder round-trips. Because reading
  v3.0 yields the exact `config.yaml` values, making it the default is
  behaviour-preserving (config.yaml remains the fallback).
- **Self-test**: new `v3.0 template faithful to config.yaml` check (CI tier).
  Full static 13/13, CI 10/10.

## [2.7.0] — 2026-06-17

### SDG config can come from the AAS (opt-in), and the SDG picker drives the passport

- **SDG tab AAS picker now takes effect**: the "Class source (AAS file)"
  selection (`fClassAas`) is sent in the run payload and used as the **template
  the new passport is cloned from** (`generate_dataset_aas` honours
  `meta['class_aas']`, falling back to `v2.0`). Picking an AAS at SDG now
  actually selects the passport's source schema.
- **SDG reads its config from the AAS** (`SmartSDG/main.py --aas_file`): when an
  AAS is supplied, `config_process.load_and_process_config` reads the SDG
  sections **from the AAS** and `config.yaml` only fills gaps (`_deep_merge`).
  New converter `aas_sdg_overrides()` extracts the faithfully-mirrored, cleanly
  typed sections — `generation_params` (scene/spawning/camera), `YOLO_CONVERSION`,
  `YOLO_FILTERING`, `SIMULATION_CONFIG` — with correct types honoured from
  `valueType`. The dashboard passes the picked AAS via `--aas_file`.
- **Scope note**: the class registry (`COMPONENT_USD_MAP`, whose USD names live
  in `valueId`) and the light tables (`LIGHT_INTENSITY_MAP` order differs;
  `LIGHT_TEMPERATURE`/`RTX` absent from the template) stay **config.yaml-
  authoritative** for now — overriding them from the current AAS would feed Isaac
  wrong values. They can be added to the override set once the template stores
  them faithfully.
- **Self-test**: new `SDG config from AAS (+ fallback)` check (CI tier) — typed
  overrides extracted from an AAS, config.yaml fills gaps. Full static 13/13.

## [2.6.0] — 2026-06-17

### AAS passport — isolated module, one record per dataset, every stage reads/writes it

- **New `aas_passport.py`** (repo root, stdlib-only so any stage's interpreter can
  import it): the single source of truth for the AAS layout. Exposes a `Passport`
  class — `Passport.for_dataset(name)` opens (or clones-from-template on first
  use) a dataset's passport; `read_config(stage)` / `write_config(stage, …)` /
  `write_measurements(stage, …)` / `append_run(…)` / `save()` let each pipeline
  stage (SDG → Training → AL → CL → Inference) read its own hyperparameters from,
  and write its parameters + measurements into, the SAME per-dataset file — the
  reproducibility "passport". Low-level helpers (`_aas_*`) moved here; the
  dashboard now imports them instead of defining its own copies.
- **Bug fix**: the template `AAS/CA_Modul-OD_pipeline_v2.0.json` had accumulated
  run residue (9 `RunHistory` entries + a stale `Results` section) from real
  runs — it is a read-only schema and must stay clean. Stripped it, and both
  `Passport.for_dataset` and the dashboard's `generate_dataset_aas` now strip
  provenance when cloning so a new passport never inherits template residue.
- **Dashboard**: the Datasets → "AAS passport" view is now a per-stage
  hyperparameter survey (SDG / Training / Active learning / Continual learning /
  Latest results) for a quick reproducibility read-out.
- **CLI self-recording**: `aas_passport.record_run(dataset, stage, config, measurements)`
  — a non-raising one-call helper. Each stage's CLI entry point now updates the
  passport on a successful run, so runs launched **outside the dashboard** are
  documented too:
  - SDG (`SmartSDG/main.py`): generation+filter config + drop stats + final image count.
  - Training (`OD_training/main.py`): training config + best-effort mAP from results.csv.
  - AL (`active_learning/run_al_from_aas.py`): AL strategy config + rounds logged.
  - CL (`continual_learning/main_cl.py`): CL config (classes, replay, epochs).
  Recording is wrapped so a passport hiccup never breaks a run. The dashboard sets
  `SMARTSDG_SKIP_PASSPORT=1` on the stage subprocess so its own `record_to_aas`
  stays the single writer (no duplicate run-history entries).
- **Self-test**: new `AAS passport module roundtrip` check (in the `--ci` tier) —
  passport + `record_run` create → write config/measurements/run → reload →
  verify; the skip flag suppresses recording; BOM-free; valid IEC 63278. Full
  static suite 12/12.

## [2.5.0] — 2026-06-16

### SDG — depth-agnostic geometric-collapse filter (containment)

- **Problem**: the overlap filter only checked **IoU**, so a small box nested
  inside a big one (e.g. a flat sliver ~90% inside another box) survived — its
  IoU is low (~0.14) because the union is dominated by the big box, and its
  `occlusionRatio` can stay under the visibility threshold.
- **`SmartSDG/src/yolo_converter.py`**: new `calculate_containment()`
  (intersection ÷ smaller-box area); `filter_overlapping_boxes()` now drops a
  frame on IoU ≥ `iou_threshold` **OR** containment ≥ `containment_threshold`.
  Depth is intentionally ignored — heavy geometric overlap is dropped regardless
  of which object is in front (visibility-based occlusion is handled earlier by
  `min_visibility`).
- **New hyperparameter** `containment_threshold` (default **0.75**), wired
  through `config.yaml` (`YOLO_FILTERING.CONTAINMENT_THRESHOLD`),
  `SmartSDG/main.py` (`--containment_threshold`), `pipeline.yaml`, and the
  dashboard (③ SDG tab field + info tooltip + command builder).
- **Occlusion handling unchanged**: any object below `min_visibility` (0.95)
  still drops the whole frame, front or behind.
- **Self-test**: new check `geometric-collapse overlap filter` (in the `--ci`
  tier) proves the nested box now drops the frame and that the legacy IoU-only
  path misses it. Full static suite 11/11.

## [2.4.0] — 2026-06-16

### CI — automated self-test on GitHub Actions

- Added `.github/workflows/ci.yml`: runs on every push / PR to `main` (and
  manually via *workflow_dispatch*).
- New `--ci` tier in `scripts/test_pipeline.py` runs the machine-independent
  subset of the self-test — guards the hard invariants (BOM-free configs,
  `config.yaml` structure, AAS template ↔ class-ID registry consistency, all
  modules compile, converter "never keep an unlabeled object" rule, IEC 63278
  AAS validity) **without** needing Isaac Sim, a GPU, torch/ultralytics, or the
  git-ignored USD assets. Installs only PyYAML + numpy + Pillow + aas-core3.0
  (~1 min). The full SDG / train / AL tiers stay local (`run_tests.bat full`).
- Status badge added to the README.
- `.ultralytics/` (machine-local ultralytics runtime cache) added to
  `.gitignore`.

### Dashboard — ③ Inference tab redesign + AL tab reorganisation

- **③ Inference tab** (was separate "③ Inference" harvest tab + "🎥 Live" tab):
  merged into one tab with a **mode picker** radio button:
  - **Images with BBoxes / labels** — pick a model and a folder of images,
    click Run Inference; every image is displayed with predicted bounding boxes
    and class labels. Uses new `/api/inference_images` endpoint (runs inline,
    returns per-image box JSON, drawn on canvas via `CLS_COLORS`).
  - **Live Camera** — source type now has three options: **Webcam** (index),
    **IP Camera** (RTSP/HTTP URL — OpenCV already handles these), **Video file**.
    IP cam URL is passed straight to `cv2.VideoCapture()`.
- **④ Active Learning tab** now contains all three AL workflow phases in order:
  - **① Harvest real-image candidates** (moved from old Inference tab): run
    model on real images, keep most-uncertain top-N with pre-labels.
  - **② Review candidates** (moved from old Inference tab): Accept/Reject grid
    with ✏ Edit; Accept moves image+labels into a `REAL_<name>` dataset.
  - **③ AL experiment** (existing): strategy picker, budget controls, gain chart.
- **`dashboard/server.py`**: new `GET /api/inference_images` endpoint — runs
  YOLO on a folder inline and returns `[{rel, boxes:[cls,cx,cy,w,h]}, ...]`.
- **`docs/GUIDE.md`**: sections 6 and 7 rewritten to document the new tab layout.

## [2.3.0] — 2026-06-16

### Docker — full cross-platform support for all Python stages

- **`docker/Dockerfile`** upgraded: single `smartsdg` image covers OD training,
  AL, CL, Inference and Dashboard; `EXPOSE 8765`; `YOLO_CONFIG_DIR` env var;
  proper layer caching.
- **`docker/entrypoint.sh`** (new): mode dispatcher — `dashboard`, `train`,
  `al`, `cl`, `inference`, `info`, `shell`, `python` passthrough.
- **`docker/docker-compose.yml`** (new): one-command startup for each stage;
  `shm_size: 2g` for PyTorch dataloader; GPU via `deploy.resources`; named
  volume for YOLO weights cache.
- **`scripts/build_docker.bat`** updated: new image name (`smartsdg`), compose usage hint.
- **Dashboard (`server.py`)**: Docker mode now injects `--shm-size=2g`,
  `SMARTSDG_WIN_ROOT` and `SMARTSDG_WIN_VAL_REAL` env vars so Windows→Linux
  path translation works automatically inside the container.
- **`OD_training/main.py`**: `_translate_win_paths()` — translates Windows
  absolute paths in the generated training config to `/workspace/...` and
  `/valdata/...` when running inside Docker (Linux).
- **`OD_training/main.py`**: `parse_sdg_metadata_from_path()` — fixed for
  Linux: `Path(win_path).name` returns the full string on Linux (backslash is
  not a separator); normalize with `.replace('\\', '/')` first.
- **`requirements.txt`**: fixed `aas-core3.0rc02` → `aas-core3.0` (wrong
  package installed as `aas_core3_rc02`, not `aas_core3`); bumped
  `ultralytics` `8.2.2` → `8.3.157` to match conda env and support YOLOv12
  (`C3k2`, `A2C2f` blocks).
- **`AAS/CA_Modul-OD_pipeline_v2.0.json`**: default training `batch_size`
  `32` → `8` (safe for 8 GB VRAM with YOLOv12l).
- **Docs**: `INSTALL.md`, `PORTABILITY.md`, `README.md` rewritten to make
  Docker the primary setup path; `GUIDE.md` updated.

## [2.2.0] — 2026-06-15

Major feature wave (all dashboard-integrated, self-test 10/10):
- **Live inference** tab (webcam/video MJPEG); pipeline tabs reordered
  SDG→Training→Inference→AL; info (ⓘ) buttons across every control; full PDF
  user guide with real screenshots.
- **Portability**: machine paths resolved via config_paths.py (env >
  pipeline_config.json > auto-detect); Docker for the Python stages.
- **Master config** `pipeline.yaml` — single source of cross-stage defaults, read
  live by the dashboard.
- **Continual Learning** fully integrated (⑤ tab): replay class-incremental, new
  classes appended, model's known classes derived from dataset name / sidecar.
- **AAS as the canonical record (A–E):** per-dataset AAS now stores generation
  parameters + training/AL/CL configs + outcome metrics + append-only RunHistory;
  a factory-wide `AAS/pipeline_history.json` (Submodel PipelineHistory) logs every
  action with session/task tags incl. inference/harvest events; all AAS files
  validated as IEC 63278 Environments (self-test check); docs/AAS_SCHEMA.md.
- Added CLAUDE.md (project memory) to reduce re-exploration cost.

## [2.1.0] — 2026-06-11

Code cleanup of `SmartSDG/` (no behavior change; smoke-tested: 3 samples,
multi-object, 640×480, all frames generated and converted identically).

### Changed
- `main.py` rewritten for clarity: module docstring with usage, CLI split into
  documented sections, logic factored into `build_arg_parser` /
  `apply_cli_overrides` / `validate_component_selection` / `print_final_summary`.
- `main.py`: `--config` default fixed from nonexistent `cl_src/config/config.yaml`
  to `src/config/config.yaml` — the flag is now optional.
- `src/yolo_converter.py`: removed fully duplicated import block and duplicate
  `_convert_coords_to_yolo`; added module docstring stating the
  never-keep-unlabeled-objects invariant.
- `src/config_process.py`: removed unnecessary imports of `generation` /
  `yolo_converter` (no functional reason); explicit imports instead of
  `from config_loader import *`; cleaned stale "NEW:" comments.
- `src/generation.py`, `src/config_loader.py`: module docstrings, tidied headers.

### Removed (moved to `SmartSDG/legacy/`, see its README)
- `src/SmartSDG.py`, `src/temp_main.py`, `src/worker_writer.py`,
  `src/yolo_writer.py` — dead code, not imported anywhere.
- `cl_src/` (root and src copies) — stale config remnants of the old layout.
- `src/__pycache__/`.

## [2.0.0] — 2026-06-11

Initial v2 working copy, cloned from the frozen v1 reference
(`C:\Users\hota02\Documents\SDG\`, do not modify).

### Code is byte-identical to v1 as of this date. Included v1 features:
- CLI resolution override (`--width`, `--height`) synced to render products and labels
- Multi-object component subsets (`--components A,B,C`) with global class IDs
- Collision-aware placement (`--min_separation`, rejection sampling)
- Spawn radius override (`--spawn_radius`)
- Quality filters with frame-level policies (never keep an image with an unlabeled object):
  - `--min_visibility` (occlusion by other objects, default 0.95, `occlusion_policy frame`)
  - `--drop_truncated` (border-cut objects, `truncation_policy frame`)
  - degenerate zero-area boxes always discard the frame
  - `--filter_overlaps --iou_threshold` (label-box IoU, default 0.7)
- FINAL DATASET SUMMARY printed at end of every run
- Renderer hyperparameters exposed in `config.yaml` (`SIMULATION_CONFIG`, `RTX_CARB_SETTINGS`)

### Packaging added in v2 (no code changes):
- Top-level README, this CHANGELOG, VERSION file
- `docs/` — INSTALL, PIPELINE, HYPERPARAMETERS guides
- Full pinned `requirements.txt`
- `scripts/run_sdg.bat` launcher pointing at SDG_v2
- Excluded from the clone: generated datasets, training runs/results,
  AL outputs, virtualenv, AASX package explorer binary

### Validated hyperparameter baseline (2026-06-10)
640×480 · min_visibility 0.95 · policies frame/frame · IoU 0.7 ·
spawn_radius 0.2 · min_separation 0.05 · light Medium · env plane

---

## Template for your next entry

## [2.x.y] — YYYY-MM-DD
### Changed
- <parameter>: <old> → <new> — <reason / observed effect on dataset & mAP>
