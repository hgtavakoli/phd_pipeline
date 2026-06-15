# AAS Schema — for modular / cross-module integration (IEC 63278)

Every config/record file in `AAS/` is a valid **Asset Administration Shell
Environment** (verified with `aas_core3`; the self-test re-checks this). Another
module can therefore parse them with any standard AAS SDK. There are two kinds.

## 1. Per-dataset record — `AAS/datasets/<date>/CA_Modul-OD_dataset_<name>.json`

One file per dataset. It is both the **input config** and the **operational
record** for that dataset. Relevant SubmodelElements (under `ODOptimisation`):

| idShort | Type | Written by | Contents |
|---|---|---|---|
| `GenerationParameters` | SubmodelElementList | SDG run | mode, components, light, env, width, height, min_visibility, spawn_radius, min_separation, iou_threshold, drop_truncated, filter_overlaps, seed, renderer, final_images, generated_at |
| training props (`base_data_dir`, `cameras_to_train`, `epochs`, `batch_size`, `img_size`, `weights`, …) | properties | Training | the OD training config used |
| `ActiveLearning` | SubmodelElementList | AL run | strategy, class_balance, rounds, initial_subset, add_per_round, replay/epochs… |
| `ContinualLearning` | SubmodelElementList | CL run | cl_strategy, input_model, current_classes, new_classes, replay_ratio, epochs… + the extended class list (`names`) |
| `Results` | SubmodelElementList | any run | latest stage, mAP50, mAP50-95, gain, model_path, epochs, updated_at |
| `RunHistory` | SubmodelElementList of SubmodelElementCollection | every run | **append-only**: one collection per completed run (stage, status, finished_at, duration_s, metrics) |

## 2. Factory event log — `AAS/pipeline_history.json`

A single AAS Environment with one Submodel **`PipelineHistory`** (id
`urn:smartsdg:submodel:pipeline-history`) → SubmodelElementList **`Events`**.
Each event is a SubmodelElementCollection of properties:

| Property | Meaning |
|---|---|
| `timestamp` | ISO time of the event |
| `session_id`, `session_name`, `task` | which session/task it belongs to |
| `stage` | SDG / training / AL / CL / inference / session |
| `action` | run_completed, harvest_completed, real_image_accepted, live_started, session_started, session_set |
| `dataset`, `model_path`, `source`, `status`, `map50`, `gain`, `duration_s`, `kept` | event-specific details |

This is the chronological, modular factory record: a higher-level system can read
`Events` to reconstruct exactly what happened across all stages and sessions.

## How to consume from another module (example)

```python
from aas_core3 import jsonization
import json
env = jsonization.environment_from_jsonable(
    json.load(open("AAS/pipeline_history.json", encoding="utf-8-sig")))
sm = next(s for s in env.submodels if s.id_short == "PipelineHistory")
events = next(e for e in sm.submodel_elements if e.id_short == "Events")
for ev in events.value:
    props = {p.id_short: p.value for p in ev.value}
    print(props["timestamp"], props["stage"], props["action"])
```

## Conventions

- Files are written **without a UTF-8 BOM** (the parsers reject BOMs).
- Property values are stored as strings (`xs:string`/`xs:float`/`xs:integer`),
  consistent with the existing templates.
- `RunHistory` / `Events` are **append-only** — nothing is overwritten, so the
  full history is preserved for audit and reproducibility.
- The per-dataset records are committed to git (versioned history); the rolling
  `pipeline_history.json` persists on disk but is git-ignored (operational log).
