# Active Learning — User Guide

Active Learning (AL) reaches a target accuracy with **far fewer labelled images**
by letting the model choose what to label next, instead of labelling everything.

This guide explains every control in the dashboard's **🧠 Active Learning** and
**🎯 AL Harvest** tabs. It is written for a professional user who is not an AL
expert — every dashboard control also has an **ⓘ** button with a short hint that
links here for the full explanation.

---

## The idea in one paragraph

You train a model on a small labelled subset (round 0). The trained model then
scores every remaining image in the pool, and a *selection strategy* picks the
most useful ones to add to the labelled set. You retrain and repeat for a few
rounds. Because the model labels its own "most informative" images first, the
accuracy curve usually climbs faster than labelling at random — so you hit your
target mAP with a fraction of the labels.

In this pipeline the **pool is your synthetic (SDG) dataset**, and **validation
is on real images**, so the gain curve directly measures sim-to-real transfer.

---

## Two ways to run AL

### 1. On synthetic data (🧠 Active Learning tab)
The selected generated dataset is treated as the unlabelled pool; AL picks the
most informative synthetic images each round. Use this to compare strategies and
camera geometries on controlled data.

### 2. On real data via inference (🎯 AL Harvest tab)
Run a trained model over a folder of real, unlabelled images. The most
*uncertain* frames are kept with the model's predictions as pre-labels. You
review/correct them in the browser (✏ Edit) and **Accept** them into a
`REAL_<name>` dataset — which then becomes trainable / AL-able like any other.
This is the human-in-the-loop loop: the model tells you which real frames are
worth labelling.

---

## Controls

### Dataset = unlabelled pool
The generated dataset whose images form the pool to select from. Generate one in
the **Run** tab first. Bigger pools give the strategy more to choose from.

### Camera geometry
SDG renders every scene from **two cameras**:
- **Replicator** — fixed top-down view
- **Replicator_01** — random orbital view

These are different viewpoints and form **separate datasets**. A model trained on
one geometry should be compared on that geometry, so pick **one camera** and run
it; the run is logged separately (`output/<dataset>__<camera>`) and never mixed
with the other. "both" merges them — only use it if you deliberately want a
combined model.

### Strategy
How the next images are chosen each round:
| Strategy | What it does | When to use |
|---|---|---|
| **random** | uniform random pick | the baseline every method must beat |
| **entropy** | most-uncertain images (model least confident) | uncertainty only |
| **ppal_box** | uncertainty + diversity by bounding-box geometry | legacy behaviour |
| **ppal_feat** | uncertainty + diversity by appearance embeddings | **recommended** |

`ppal_feat` uses the detector's own feature embeddings to avoid selecting many
look-alike images — the most faithful to the PPAL paper and usually the best.

### Class-balance boost
Real pools are often class-imbalanced. With this on, images containing
under-represented classes get a selection bonus, filling in rare classes faster.

### Budget: rounds / initial subset / add per round
- **Initial subset** — images labelled and trained on before the first selection
  (round 0 seed set). Too small and the first model is too weak to score well; a
  few dozen is typical.
- **AL rounds** — how many select+retrain cycles follow round 0.
- **Add per round** — images added each round (the labelling budget per step).

Constraint: `initial + rounds × add` must not exceed the pool size.

### Training: epochs / batch / image size / initial weights
- **Epochs per round** — 3–10 for quick experiments, 50+ for real results.
- **Batch** — larger is faster but uses more GPU memory; on an 8 GB RTX 2080 use
  8–16 at 640px with YOLOv12-L, or `-1` for auto-fit.
- **Initial weights** — the pretrained checkpoint round 0 fine-tunes from;
  **YOLOv12** is the project default.

---

## Reading the Gain Overview

Each line is one AL run (dataset + camera + strategy), plotting **mAP50 vs number
of labelled images**. The **gain** column is *best mAP50 − round-0 mAP50*.

To show AL works, run the **same dataset + camera** with `random` and with
`ppal_feat`, then compare the two lines at the same number of labelled images —
the vertical gap is exactly what Active Learning bought you. Compare the two
camera geometries as separate lines to see which viewpoint transfers better.

---

## Where everything is saved

| What | Path |
|---|---|
| AL run outputs + learning curve | `active_learning/output/<dataset>[__<camera>]/al_summary.csv` |
| Harvest candidates (pending review) | `active_learning/harvest/<name>/{images,labels}` |
| Accepted real dataset | `SmartSDG/Datasets/yolo_experiments/REAL_<name>/` |
| AL parameters (record) | the dataset's AAS file under `AAS/datasets/<date>/` |

Reproducibility: every AL run writes its parameters into the dataset's AAS file
(record mode), so the config and the dataset always travel together.
