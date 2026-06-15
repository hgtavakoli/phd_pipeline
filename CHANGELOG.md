# Changelog

All notable changes to this pipeline are documented here.
Format: [Keep a Changelog](https://keepachangelog.com). Record **every**
hyperparameter change — SDG results are highly sensitive to them.

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
