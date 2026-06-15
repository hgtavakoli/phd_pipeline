# SmartSDG Pipeline — Comprehensive User Guide

> A complete, example-driven manual for the whole pipeline, written for a
> professional user who is **not** expected to be an expert in synthetic data,
> object detection, or active learning. Every control in the dashboard has an
> **ⓘ** button whose "Open full guide" link brings you here.

---

## Table of contents

1. [What this pipeline does](#1-what-this-pipeline-does)
2. [The big picture (diagram)](#2-the-big-picture)
3. [Starting the dashboard](#3-starting-the-dashboard)
4. [Step ① — Synthetic Data Generation (SDG)](#4-step--synthetic-data-generation)
5. [Step ② — Object Detection Training](#5-step--object-detection-training)
6. [Step ③ — Inference & harvesting real candidates](#6-step--inference--harvesting)
7. [Step ④ — Active Learning](#7-step--active-learning)
8. [Labelling & the box editor](#8-labelling--the-box-editor)
9. [Where everything is stored](#9-where-everything-is-stored)
10. [Reproducibility & the AAS record](#10-reproducibility--the-aas-record)
11. [Worked end-to-end example](#11-worked-end-to-end-example)
12. [Troubleshooting](#12-troubleshooting)
13. [Glossary](#13-glossary)

---

## 1. What this pipeline does

You want an object detector that works on **real** images of industrial truck
components — but labelling thousands of real photos by hand is expensive. This
pipeline lets you:

1. **Generate** perfectly-labelled training images in a 3D simulator (no photos,
   no manual labelling).
2. **Train** a YOLOv12 detector on them.
3. **Run** that detector on real images to find the ones it struggles with.
4. **Actively learn**: label only those most-useful images and retrain, reaching
   high accuracy with a fraction of the labelling effort.

Everything is driven from one local dashboard and recorded in AAS config files
so any experiment is reproducible.

---

## 2. The big picture

```
            ┌──────────────────────────────────────────────────────────┐
            │                     SmartSDG Pipeline                      │
            └──────────────────────────────────────────────────────────┘

   ① SDG                ② TRAINING            ③ INFERENCE         ④ ACTIVE LEARNING
 ┌─────────┐          ┌────────────┐        ┌────────────┐      ┌────────────────┐
 │ Isaac   │  YOLO    │  YOLOv12   │ best.pt│  run on    │ hard │  select most   │
 │ Sim     │ dataset  │  training  │  ────► │  real imgs │ imgs │  useful images │
 │ renders │ ───────► │ per camera │        │  (uncertain│ ───► │  round by round│
 │ + labels│          │            │        │  harvest)  │      │  → retrain     │
 └─────────┘          └────────────┘        └────────────┘      └────────────────┘
      │                     │                     │                     │
      ▼                     ▼                     ▼                     ▼
 yolo_experiments/    results/yolo_run/    harvest/ + REAL_    output/<ds>/al_summary.csv
                                            datasets             (gain curve)

         every step's settings are saved into an AAS .json (the record)
```

Each scene is rendered by **two cameras** — `Replicator` (fixed top-down) and
`Replicator_01` (random orbital) — and these are kept as **separate datasets**
throughout, so you can train and compare each viewpoint independently.

---

## 3. Starting the dashboard

```bat
:: activate the training environment first, then:
scripts\launch_dashboard.bat
```

This opens `http://127.0.0.1:8765`. The top bar shows a **status badge**
(idle / running / finished). The tabs follow the pipeline order:

```
 ① SDG | ② Training | ③ Inference | ④ Active Learning | Console | Datasets | Results | Docs
```

Any long-running job streams its output live to the **Console** tab.

---

## 4. Step ① — Synthetic Data Generation

**Goal:** produce a labelled YOLO dataset.

### Controls
- **Class source** — the AAS file whose class list defines the objects and their
  global class IDs. To add a new object: add its name here and drop `<Name>.usd`
  into `SmartSDG/assets/Incorap_lkw/`; it auto-registers with the next free ID.
- **Components** — tick the objects to render. Grayed = missing USD file.
- **Mode** — `multi-object` (several parts per scene, for detection) or
  `single-object`.
- **Light / Images / Environment / Resolution / Seed** — scene settings. Seed
  makes a run fully reproducible (same object positions, rotations, lighting).
- **Quality filters** — min-visibility, spawn radius, min-separation, IoU
  overlap, drop border-cut. These guarantee every kept image contains only
  fully-visible, fully-labelled, non-overlapping objects (see
  [HYPERPARAMETERS.md](HYPERPARAMETERS.md)).
- **Advanced renderer** — samples-per-pixel, bounces, denoiser (render quality).

### Example
Generate 100 multi-object images of three parts at 640×480 on a plane:

```
Components: Cabin, GlueGun, BLid
Mode: multi-object   Light: Medium   Images: 100   Environment: plane
Resolution: 640×480  Min visibility: ≥95%  Spawn 0.2  Separation 0.05
```

Press **▶ Run generation**. The Console ends with:

```
##              FINAL DATASET SUMMARY                        ##
  Frames rendered by Isaac Sim       : 200
  Frames discarded - object occluded < 95% visible : 31
  FINAL images in dataset            : 142
  Output: ...\yolo_experiments\MULTI_Cabin-GlueGun-BLid_Medium_plane
```

The dataset now appears in the **Datasets** tab (with thumbnails + box preview).

---

## 5. Step ② — Object Detection Training

**Goal:** train a YOLOv12 detector on a generated dataset.

- **AAS config** — the dataset's config (auto-created after generation, listed
  at the top of the dropdown).
- **Cameras** — `Replicator` and/or `Replicator_01`. **Each ticked camera trains
  as its own separate run** — the two viewpoints never merge.
- **Epochs / Image size / Batch / Base weights / Save period** — standard
  training knobs. Empty fields use the AAS values; filled fields are saved back
  into the dataset's AAS (**record mode**).

> **Memory tip:** on an 8 GB RTX 2080, batch 32 at 640px with YOLOv12-L can run
> out of memory — use batch 8–16, or `-1` for auto-fit.

Press **▶ Run training**. Results (weights `best.pt`, PR/F1 curves, confusion
matrices) land in `OD_training/results/yolo_run/exp{0,1,2}/<dataset>/` and show
up in the **Results** tab, one row per camera run, with a Weights ✓/— column.

---

## 6. Step ③ — Inference & harvesting

**Goal:** use a trained model to find the **real** images most worth labelling.

1. **Model** — pick a `best.pt` from your training runs.
2. **Real images folder** — click **📁 Browse…** to navigate your drives and pick
   the folder (no need to type the path).
3. **Keep top-N uncertain** — how many of the hardest frames to keep.
4. **Conf threshold** — detection confidence used while scoring.

Press **▶ Harvest**. The model scores every real image by uncertainty and keeps
the top-N, saving each with its predicted boxes as **pre-labels**. They appear in
the **Review candidates** grid with boxes drawn.

For each candidate:
- **✏ Edit** — open the box editor to correct the labels (see §8).
- **✔ Accept** — moves the image + labels into a `REAL_<name>` dataset (then
  trainable / AL-able like any dataset).
- **✖ Reject** — discards it.

This is the human-in-the-loop: the model tells you which real frames are worth
your labelling time.

---

## 7. Step ④ — Active Learning

**Goal:** reach a target accuracy with far fewer labelled images, by letting the
model choose what to label next. Full details in
[ACTIVE_LEARNING.md](ACTIVE_LEARNING.md).

- **Dataset** = the unlabelled pool (a generated dataset).
- **Camera geometry** — run one camera at a time; each is logged separately so
  the two compare independently.
- **Strategy** — `random` (baseline) / `entropy` / `ppal_box` / `ppal_feat`
  (recommended).
- **Class-balance** — boosts rare classes.
- **Budget** — initial subset, rounds, add-per-round.

Each round trains a model, scores the pool, and adds the selected images. The
**Gain overview** chart plots **mAP50 vs number of labelled images** for every
run. To prove AL works, run the same dataset+camera with `random` and with
`ppal_feat` and compare the two lines — the gap is the gain.

---

## 8. Labelling & the box editor

Click **✏ Edit** on any harvest candidate or dataset thumbnail to open the
editor:

| Action | How |
|---|---|
| New box | drag on an empty area |
| Select | click a box |
| Move | drag inside a selected box |
| Resize | drag the bottom-right corner handle |
| Delete | Del key, or the delete button |
| Change class | the class dropdown (applies to the selected/new box) |

**💾 Save** writes standard YOLO labels (`class cx cy w h`, normalised). Files are
plain text, so you can also refine them in LabelImg / Roboflow / CVAT.

---

## 9. Where everything is stored

| What | Path |
|---|---|
| Synthetic dataset | `SmartSDG/Datasets/yolo_experiments/<name>/` |
| Real (accepted) dataset | `SmartSDG/Datasets/yolo_experiments/REAL_<name>/` |
| Raw render (pre-conversion) | `SmartSDG/Datasets/unified_render_output/<name>/` |
| Training runs | `OD_training/results/yolo_run/exp{0,1,2}/<name>/` |
| Harvest candidates | `active_learning/harvest/<name>/{images,labels}` |
| AL outputs + gain curve | `active_learning/output/<name>[__<camera>]/al_summary.csv` |
| Config record (per dataset) | `AAS/datasets/<date>/CA_Modul-OD_dataset_<name>.json` |

`exp0` = void background, `exp1` = plane, `exp2` = full 3D environment.

---

## 10. Reproducibility & the AAS record

Every dataset gets an **AAS** (Asset Administration Shell, IEC 63278) JSON file
that records how it was generated and trained. The dashboard runs in **record
mode**: when you change training or AL parameters, they are written into that
file, so the dataset and the exact configuration that produced each result
always travel together. These files are committed to git, giving a full history
of every parameter change (`git diff`).

For exact run reproduction, set a **seed** in SDG — the scene geometry and labels
become byte-identical across runs (rendered pixels match up to GPU floating-point
noise, the standard caveat for any GPU renderer).

---

## 11. Worked end-to-end example

A complete cycle, all from the dashboard:

```
① SDG:        generate 100 images (Cabin, GlueGun, BLid), Medium, plane, 640×480
              → dataset MULTI_Cabin-GlueGun-BLid_Medium_plane (≈142 imgs after filtering)

② Training:   AAS = that dataset, camera = Replicator only, 50 epochs, batch 8,
              weights yolov12l.pt → best.pt in results/.../exp1/...

③ Inference:  model = that best.pt, Browse to your real-images folder,
              keep top 50, conf 0.10 → Harvest
              → review the 50 hardest real frames, Edit/Accept the good ones
              → creates dataset REAL_<folder>

④ AL:         dataset = the synthetic pool, camera = Replicator,
              strategy ppal_feat, 3 rounds, init 10, add 10
              → run again with strategy random
              → Gain overview shows ppal_feat above random = AL works
```

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: aas_core3` (training) | activate the `yolov12` env before launching the dashboard, or `pip install aas-core3.0rc02` |
| Generation ends with `No valid data folders` | a config/parse error — run **Pipeline Self-Test** (Quick) to pinpoint it |
| `ParserError ... expected '<document start>'` | a config or AAS file has a UTF-8 BOM; the self-test "no BOM" check flags it |
| Training "Skipping … Path not found" | the AAS `base_data_dir` / camera names don't match the dataset folders |
| Out of GPU memory | lower batch size, or use `-1` for auto-fit; reduce image size |
| Dashboard shows "server offline" | the local server stopped — relaunch `scripts\launch_dashboard.bat` |

Run `scripts\run_tests.bat` (or the **Pipeline Self-Test** buttons) after any
change — it checks configs, assets, the class-ID registry, converter filters,
exp routing, and (full mode) a real generation, training and AL loop.

---

## 13. Glossary

- **SDG** — Synthetic Data Generation (rendering labelled images in a simulator).
- **YOLO label** — a `.txt` per image; each line `class cx cy w h` (normalised).
- **Class ID** — the integer for each component, fixed by the registry order in
  `config.yaml`; never reordered so old datasets stay valid.
- **AAS** — Asset Administration Shell; the JSON config/record per dataset.
- **Camera geometry** — Replicator (top-down) vs Replicator_01 (orbital); kept
  separate.
- **Active Learning (AL)** — iteratively labelling the most informative images.
- **mAP50 / mAP50-95** — mean Average Precision, the standard detection accuracy
  metrics.
- **Uncertainty / diversity** — the two ingredients of AL selection: how unsure
  the model is, and how different the chosen images are from each other.
- **Record mode** — the dashboard saves your parameters into the dataset's AAS.
