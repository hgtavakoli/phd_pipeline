# CLAUDE.md — project memory for SmartSDG Pipeline v2

Read this first. It captures everything needed to work here WITHOUT re-exploring
the codebase, so keep edits surgical and avoid re-reading large files.

## What this is
End-to-end synthetic-data → detection pipeline, dashboard-driven, AAS-recorded.
Stages: ① SDG (Isaac Sim) → ② OD Training (YOLOv12) → ③ Inference/harvest →
④ Active Learning → ⑤ Continual Learning. One local dashboard orchestrates all.

## Environments / paths (this machine)
- Training/dashboard Python: `C:\Users\hota02\AppData\Local\anaconda3\envs\yolov12\python.exe` (conda env `yolov12`; has torch+ultralytics+aas_core3+opencv+markdown)
- Isaac Sim Python (SDG only): `C:\Users\hota02\Documents\omniverse\python.bat`
- These are auto-resolved by `config_paths.py` (env var > pipeline_config.json > auto-detect). Never hardcode again.
- Dashboard server: `dashboard/server.py`, serves http://127.0.0.1:8765 (stdlib only, single job at a time).

## Run / verify
- Dashboard: `scripts\launch_dashboard.bat` (or `python dashboard/server.py`).
- Self-test (use this to verify instead of manual exploration): `scripts\run_tests.bat` (static, ~12s, 10 checks) or `... full` (adds real SDG+train+AL). All must stay green.
- Restart server pattern (PowerShell): kill `Get-NetTCPConnection -LocalPort 8765` owning process, then Start-Process the server hidden.

## Config layers (don't merge these — by design)
- `pipeline.yaml` = cross-stage DEFAULTS the dashboard pre-fills (served live via /api/defaults).
- `SmartSDG/src/config/config.yaml` = SDG params + `COMPONENT_USD_MAP` (the class registry; **append-only**, global IDs never reorder; Isaac reads it directly).
- `AAS/datasets/<date>/CA_Modul-OD_dataset_<name>.json` = per-dataset record (config + outcomes), git-committed.
- `AAS/pipeline_history.json` = factory event log (git-ignored, rolling).
- `pipeline_config.json` = per-machine paths (git-ignored).

## Hard invariants (do not break)
- **AAS files must be BOM-free** (Isaac yaml + aas_core3 reject BOMs). Always write with UTF-8 no-BOM. PowerShell `Set-Content -Encoding utf8` adds a BOM — use `[IO.File]::WriteAllText(p, s, (New-Object Text.UTF8Encoding $false))` or the Write tool.
- **Never keep an image with an unlabeled object** — occluded/truncated/degenerate objects discard the WHOLE frame (yolo_converter policies default "frame").
- **Class registry append-only** — new components append to COMPONENT_USD_MAP at the end (next free ID); existing IDs never shift.
- **Record mode**: dashboard writes run params + outcomes into the per-dataset AAS; every completion also appends to `pipeline_history.json`. Keep both.
- Validated SDG defaults: 640×480, min_visibility 0.95, spawn 0.2, separation 0.05, IoU 0.7, drop_truncated, light Medium, env plane.
- Default model is **YOLOv12** (yolov12l.pt) everywhere user-facing.

## Stage specifics
- AL strategies: random | entropy | ppal_box | ppal_feat (default, embedding diversity). Per-camera (Replicator/Replicator_01) kept separate. Gain logged to al_summary.csv.
- CL: replay-based class-incremental. New-class data is global-ID→0-based remapped before main_cl.py's offset. Model's trained classes derived from dataset-name in path (or trained_classes.json sidecar) — NOT model.names (always full registry).
- Live inference: MJPEG `/stream`; `/api/stop_stream` releases the camera (token bump); single stream only (concurrent webcam open hard-crashes on Windows).

## Conventions
- Commit only when asked; end commit body with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Use a `.git\CMTMSG.txt` file + `git commit -F` for multi-line messages (avoids PowerShell quoting breakage).
- Background long jobs; verify via the self-test or a single targeted check, not broad re-reads.
- There are usually unpushed commits; user runs `git push` themselves.

## Docs map (point the user here, don't re-summarize)
`docs/GUIDE.md` (full guide+PDF), `docs/CONFIG.md`, `docs/AAS_SCHEMA.md`,
`docs/ACTIVE_LEARNING.md`, `docs/CONTINUAL_LEARNING.md`, `docs/HYPERPARAMETERS.md`,
`docs/PORTABILITY.md`, `docs/INSTALL.md`, `docs/PIPELINE.md`.
