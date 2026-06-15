# Continual Learning — User Guide

Continual Learning (CL) lets you **add a new object class to an already-trained
model** without retraining from scratch and without the model forgetting the
classes it already knew (*catastrophic forgetting*). This is the **⑤ Continual
Learning** tab in the dashboard.

---

## When to use CL vs. retraining

| | Full retrain (Training tab) | Continual Learning (this tab) |
|---|---|---|
| New class added | retrain on ALL classes from scratch | extend the existing model |
| Data needed | all classes' data | mainly the new class + a replay sample of old |
| Time / compute | high | low |
| Risk | none (sees everything) | must guard against forgetting old classes |

Use CL when you already have a good model and want to grow it by one or a few
classes cheaply. Use the comparison table at the bottom of the CL tab to verify
the trade-off on your data.

---

## How it works (the method)

This pipeline uses **replay-based (rehearsal) class-incremental learning**:

1. Start from your trained model (it knows classes `0..N-1`).
2. The new class becomes the next global ID `N`.
3. Each training batch is **balanced**: a fixed fraction (the *replay ratio*) is
   old-class images, the rest is new-class images. Rehearsing the old classes
   stops the model from forgetting them while it learns the new one.

The class IDs stay consistent with the rest of the pipeline: the dashboard
remaps the new-class data automatically so the new class lands at the correct
global ID `N`, continuing the append-only registry.

> **Where the field is going:** replay is a strong, standard baseline. The SOTA
> next step for incremental object detection is adding **knowledge distillation**
> (a frozen copy of the old model teaches the new one on each image). The CL tab
> has a `strategy` selector so distillation can be added alongside `replay`
> later without changing the workflow.

---

## Controls

| Control | Meaning |
|---|---|
| **Base model to extend** | the `best.pt` whose classes you keep and build on |
| **Strategy** | `replay` (balanced rehearsal). More strategies can be added. |
| **Replay ratio** | fraction of each batch that is old-class data (e.g. 0.3 = 30%). Higher protects old classes more, learns the new one slower. |
| **New class(es)** | which component(s) to add — must be the next class(es) after the model's existing ones |
| **New-class data** | dataset containing the new class. Auto-matched from the pool by name; if missing, generate it in the SDG tab first |
| **Replay data** | dataset of the OLD classes to rehearse (usually what the model was trained on) |
| **Camera** | geometry filter, to match the base model's geometry |
| **Epochs / Batch / Image size** | training settings for this step (usually fewer epochs than a full train) |

---

## If the new-class data isn't in the pool yet

The CL tab checks whether a generated dataset for the new class already exists:

- **It does** → the dataset is auto-selected; just run.
- **It doesn't** → Run CL offers to take you to the SDG tab to generate it.
  Generate a dataset for the new component, then come back and run CL on it.

This is the "generate-on-demand" path: synthetic data for the new class is
produced first, then the model is incrementally trained on it.

---

## Overview, evaluation & comparison

The bottom card lists every CL run with its final **mAP50 / mAP50-95**, epoch
count, and whether weights were produced. To judge CL properly:

- **Old-class retention:** validate the new model on the old classes (Results /
  the run's validation) — it should stay close to the pre-CL accuracy.
- **New-class accuracy:** the new class should be learned well.
- **Cost vs. full retrain:** compare the CL run's time/epochs against training a
  fresh model on all classes (Training tab). CL should be markedly cheaper.

The console prints the **batch composition** (new vs replay per batch) and
per-epoch timing for each run.

---

## Where CL outputs are saved

| What | Path |
|---|---|
| CL run (weights, curves) | `continual_learning/results/cl_runs/<exp_name>/` |
| Staged new/replay data (temp) | `continual_learning/data/` |
| CL parameters (record) | the dataset's AAS, `ContinualLearning` section |

After a successful step the extended model's new class list is recorded so the
dataset + the exact CL configuration travel together (reproducibility).
