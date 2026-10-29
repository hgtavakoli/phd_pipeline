# Prompt lab — model `llama3:latest`, temperature 0.0

## ❓ What does the SDG stage do?

### `v4_fewshot`  · prompt 746 tok · output 34 tok

The SDG stage, also known as the Synthetic Data Generation stage, uses NVIDIA Isaac Sim to render scenes of industrial components and converts them to YOLO labels.

### `v5_German.md`  · prompt 746 tok · output 34 tok

The SDG stage, also known as the Synthetic Data Generation stage, uses NVIDIA Isaac Sim to render scenes of industrial components and converts them to YOLO labels.

## ❓ Which active learning strategy is the default?

### `v4_fewshot`  · prompt 746 tok · output 45 tok

The default active learning strategy is ppal_feat, which selects images by embedding diversity. The other options are random, entropy and ppal_box.

Where: ④ Active Learning tab; results in al_summary.csv.

### `v5_German.md`  · prompt 746 tok · output 45 tok

The default active learning strategy is ppal_feat, which selects images by embedding diversity. The other options are random, entropy and ppal_box.
Where: ④ Active Learning tab; results in al_summary.csv.

## ❓ What resolution should I use for synthetic images?

### `v4_fewshot`  · prompt 747 tok · output 20 tok

According to the facts, the validated SDG defaults include a resolution of 640x480.

### `v5_German.md`  · prompt 747 tok · output 20 tok

According to the facts, the validated SDG defaults include a resolution of 640x480.

## ❓ Where is the trained model saved after training?

### `v4_fewshot`  · prompt 747 tok · output 35 tok

The trained model is saved in the OD_training/results/yolo_run/exp{N}/<dataset>/<camera>/weights/best.pt, where N is the experiment number.

### `v5_German.md`  · prompt 747 tok · output 35 tok

The trained model is saved in the OD_training/results/yolo_run/exp{N}/<dataset>/<camera>/weights/best.pt, where N is the experiment number.

## ❓ How many GPUs did we use for the experiments in the thesis?

### `v4_fewshot`  · prompt 751 tok · output 46 tok

I don't know that about this project. The number of GPUs used for the experiments in the thesis is not mentioned in the <facts> section. You can check the docs/ folder or the Docs tab for more information.

### `v5_German.md`  · prompt 751 tok · output 46 tok

I don't know that about this project. The number of GPUs used for the experiments in the thesis is not mentioned in the <facts> section. You can check the docs/ folder or the Docs tab for more information.

## ❓ What is an AAS passport?

### `v4_fewshot`  · prompt 745 tok · output 56 tok

An AAS passport is a file that stores per-stage configuration, measurements, and an append-only RunHistory for every dataset in the SmartSDG pipeline. It is stored in the AAS/datasets/<date>/CA_Modul-OD_dataset_<name>.json file.

### `v5_German.md`  · prompt 745 tok · output 56 tok

An AAS passport is a file that stores per-stage configuration, measurements, and an append-only RunHistory for every dataset in the SmartSDG pipeline. It is stored in the AAS/datasets/<date>/CA_Modul-OD_dataset_<name>.json file.
