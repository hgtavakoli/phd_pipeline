# Configuration — what lives where

The pipeline deliberately uses a small, layered set of config sources rather
than one giant file, so that cross-stage defaults, the class registry, per-machine
paths, and per-dataset records each live where they belong. There are **four**:

| File | Scope | Edit when… | In git? |
|---|---|---|---|
| **`pipeline.yaml`** | Cross-stage **defaults** the dashboard pre-fills (SDG, Training, AL, CL, inference) + a mirror of the class list | you want to change the default values shown in the dashboard for any stage | ✅ |
| **`SmartSDG/src/config/config.yaml`** | SDG scene/camera/renderer params **and the authoritative `COMPONENT_USD_MAP`** (defines global class IDs; read directly by Isaac Sim) | you change SDG rendering internals, or add a component (auto-appended) | ✅ |
| **`AAS/datasets/<date>/*.json`** | The **per-dataset record** — the training/AL/CL parameters actually used for that dataset (record mode) | never by hand normally; the dashboard writes these | ✅ |
| **`pipeline_config.json`** | **Per-machine paths** (Isaac python, training python, val images) | once per machine (or use env vars / auto-detect) | ❌ (git-ignored) |

## Why not one single file?

Because the four serve genuinely different roles:

- **Defaults** are global and committed → `pipeline.yaml`.
- **The class registry** must be read by Isaac Sim's *separate* Python and is
  append-only (IDs never shift) → stays in `config.yaml`.
- **Per-dataset records** are the reproducibility mechanism (AAS / IEC 63278):
  each dataset carries the exact config that produced its results, with git
  history. Folding these into one global file would destroy per-dataset history.
- **Machine paths** can't be one committed value (they differ per computer).

So `pipeline.yaml` is the single place for *defaults*; the AAS files remain the
single source of truth *per dataset*. Changing a default in `pipeline.yaml`
updates the dashboard form for every stage (it is read live via `/api/defaults`),
but the value actually used by a run is still what the dashboard sends and the
AAS records — preserving reproducibility.

## How a value flows

```
pipeline.yaml (default)  ->  dashboard form field  ->  user may override  ->
   run command  ->  written into the dataset AAS (record)  ->  result
```

## Legacy / unused configs

`active_learning/legacy/config_yolo.py` and the CLI-only YAMLs under
`OD_training/src/configs/` are **not** used by the dashboard/AAS flow. They are
kept for reference only.
