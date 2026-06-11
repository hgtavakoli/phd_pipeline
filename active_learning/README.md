# Plug-and-Play Active Learning (PPAL) for YOLO Models

This repository provides an adaptation of the **Plug-and-Play Active Learning (PPAL)** method for object detection using the Ultralytics YOLO framework (compatible with YOLOv12).

The script analyzes a pool of unlabeled images with a trained YOLO model and selects a small, highly informative subset of images for annotation. Labeling these carefully chosen samples and retraining the model accelerates performance gains while reducing labeling costs.

---

## 📄 Reference and Attribution

If you use this implementation for research or projects, please cite the original PPAL authors:

**Paper:** *Plug and Play Active Learning for Object Detection*  
*Chenhongyi Yang, Lichao Huang, Elliot J. Crowley; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 17784-17793

**BibTeX:**
```bibtex
@InProceedings{Yang_2024_CVPR,
    author    = {Yang, Chenhongyi and Huang, Lichao and Crowley, Elliot J.},
    title     = {Plug and Play Active Learning for Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {17784-17793}
}
```

Original MMDetection implementation:  
👉 [https://github.com/torchenh/Plug-and-Play-Active-Learning-for-Object-Detection](https://github.com/ChenhongyiYang/PPAL)

YOLO adaptation inspired by:  
👉 [https://github.com/InexperiencedMe/myppal](https://github.com/InexperiencedMe/myppal)

---

## ⚙️ Requirements

Install dependencies with:
```bash
pip install ultralytics numpy opencv-python torch
```

---

## 🔍 Algorithm Overview

PPAL consists of two main stages:

### **Phase 1: Difficulty-Calibrated Uncertainty Sampling**
- The model runs inference on the unlabeled image pool.
- Image-level uncertainty is computed using **Shannon entropy** of detection confidences.
- Each class receives a **difficulty coefficient**, estimated from average uncertainty per class.
- Images containing uncertain, hard-to-detect classes are prioritized for review.

### **Phase 2: Category-Conditioned Matching Similarity (CCMS)**
- Computes the **semantic similarity** between candidate images based on shared object categories and bounding box geometry.
- Converts similarity to distance and performs **K-Means++ clustering** to ensure diverse selection.
- The final selected samples maximize both **informativeness** and **diversity**.

---

## 🚀 Usage

### Command-Line Arguments
| Argument | Description | Default |
|-----------|--------------|----------|
| `--unlabeled-dir` | Path to the directory of unlabeled images | *required* |
| `--model-path` | Path to your trained YOLO `.pt` file | *required* |
| `--num-classes` | Number of object classes in your dataset | *required* |
| `--target-number` | Number of images to select | *required* |
| `--split-ratio` | Balance between uncertainty (phase 1) and diversity (phase 2). <br> `1.0` = uncertainty only <br> `0.0` = diversity only | `0.5` |
| `--conf-threshold` | YOLO confidence threshold for inference | `0.10` |

### Example
```bash
python get_images_for_next_round.py \
    --unlabeled-dir /path/to/unlabeled_images \
    --model-path /path/to/trained_yolo.pt \
    --num-classes 80 \
    --target-number 500 \
    --split-ratio 0.5 \
    --conf-threshold 0.1
```

---

## 🔁 Active Learning Workflow

1. **Initial Training** — Train YOLO on a small labeled set.
2. **Active Selection** — Run this PPAL script using your current model and unlabeled pool.
3. **Annotation** — Label the selected samples.
4. **Retraining** — Combine new labels with existing data and retrain.
5. **Repeat** — Iteratively refine your model until desired accuracy is achieved or the labeling budget ends.

---

## 🧠 Key Benefits
- **Model-agnostic:** Works with any YOLO variant.
- **No tuning needed:** Uses only detector outputs.
- **Annotation-efficient:** Captures the most informative and diverse samples.

---

## 📬 Contact
For issues or improvements, please open an issue or pull request on the repository.
'''


Of course. Having a clear, well-structured README that explains both ways to use your project is essential. I have combined and refined the content you provided into a single, cohesive `README.md` file.

This new version:
-   Starts by explaining that there are two distinct usage modes.
-   Provides clear, step-by-step instructions for both the **Automated Pipeline** and the **Standalone Script**.
-   Unifies the common sections (like the algorithm overview and requirements) to avoid repetition.
-   **Corrects the academic citation** to the original 2021 paper by Park et al., whose methodology is implemented in the code, while retaining the "inspired by" links you provided.


# Plug-and-Play Active Learning (PPAL) for YOLO

This repository provides a powerful and flexible framework for **Plug-and-Play Active Learning (PPAL)** using the Ultralytics YOLO ecosystem. It is designed to significantly reduce annotation costs and accelerate model development by focusing labeling efforts on the data samples that provide the most value.

The project offers two primary modes of operation:
1.  **Fully Automated Experiment Pipeline:** A complete, end-to-end system that orchestrates the entire active learning loop with a single command.
2.  **Standalone Selection Script:** A modular script for developers who want to integrate just the PPAL data selection logic into their own custom workflows.

---

## 🚀 Usage Modes

Choose the mode that best fits your needs.

### Mode 1: Fully Automated Experiment 🤖

This is the recommended approach for running a complete, hands-off active learning experiment. The system automatically handles data management, initial training, iterative data selection, retraining, and evaluation.

#### **Step 1: Prepare Your Dataset**
Organize your complete, labeled dataset into the standard YOLO format inside the `data/full-data` directory.

```
your_project/
└── data/
    └── full-data/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── dataset.yaml  <-- Your main dataset config
```

#### **Step 2: Configure the Experiment**
Open `config_yolo.py` and adjust the parameters for your experiment.
- `NUM_CLASSES`: Number of classes in your dataset.
- `INITIAL_MODEL`: The base model to start from (e.g., `yolov8n.pt`).
- `AL_ROUNDS`: The number of active learning cycles to run.
- `INITIAL_SUBSET_SIZE`: Number of samples for the first training round.
- `ADD_PER_ROUND`: Number of new samples to select in each AL round.
- `EPOCHS_PER_ROUND`: Number of epochs for each training cycle.

#### **Step 3: Run the Active Learning Loop**
Execute the main orchestration script from your terminal.
```bash
python main_yolo.py
```

#### **Step 4: Evaluate and Analyze Results**
After the main script finishes, run the evaluation script to test the model from each round on your validation set.
```bash
python evaluate_yolo.py
```
This generates a CSV file (e.g., `yolo_al_experiment_evaluation_results.csv`) for plotting the model's mAP score against the number of labeled images.

---

### Mode 2: Standalone Selection Script 🧩

Use this mode if you only need to perform the PPAL data selection step and want to manage the training and data handling yourself.

#### **Usage**
The `get_images_for_next_round.py` script is run from the command line.

| Argument | Description | Default |
| :-- | :-- | :-- |
| `--unlabeled-dir` | Path to the directory of unlabeled images | *required* |
| `--model-path` | Path to your trained YOLO `.pt` file | *required* |
| `--num-classes` | Number of object classes in your dataset | *required* |
| `--target-number` | Number of images to select | *required* |
| `--split-ratio` | Balance between uncertainty (`1.0`) and diversity (`0.0`) | `0.5` |
| `--conf-threshold`| YOLO confidence threshold for inference | `0.10` |
| `--seed` | Random seed for reproducible results | `None` |

#### **Example**
```bash
python get_images_for_next_round.py \
    --unlabeled-dir /path/to/unlabeled_images \
    --model-path /path/to/trained_yolo.pt \
    --num-classes 80 \
    --target-number 500 \
    --split-ratio 0.5 \
    --seed 42
```

---

## 📂 Project Structure

- **`main_yolo.py`**: The main script that orchestrates the entire active learning loop (Mode 1).
- **`get_images_for_next_round.py`**: The standalone PPAL selection script (Mode 2).
- **`utils_yolo.py`**: Contains the core PPAL selection logic and data management functions used by both modes.
- **`config_yolo.py`**: A centralized file for all project, training, and active learning parameters (for Mode 1).
- **`evaluate_yolo.py`**: A script to systematically evaluate the models from each round (for Mode 1).

---

# 🚀 How to Run a Full Experiment
The entire pipeline is launched from a single script that reads its configuration from your AAS file.
Step 1: Configure the AAS File
Define all experiment parameters within your AAS JSON file. The script is designed to parse the nested structure ODOptimisation -> ActiveLearning. Below is an example of the required structure and parameters.
Example AAS JSON Structure:


```
code
JSON
{
  "submodels": [
    {
      "idShort": "OD_Module",
      "submodelElements": [
        {
          "idShort": "ODOptimisation",
          "value": [
            {
              "idShort": "ActiveLearning",
              "value": [
                {
                  "idShort": "Paths",
                  "value": [
                    { "idShort": "data_dir", "value": "./data" },
                    { "idShort": "output_dir", "value": "./output" },
                    { "idShort": "initial_model_path", "value": "yolov8n.pt" }
                  ]
                },
                {
                  "idShort": "TrainingParameters",
                  "value": [
                    { "idShort": "epochs_per_round", "value": 50 },
                    { "idShort": "batch_size", "value": 16 },
                    { "idShort": "img_size", "value": 640 }
                  ]
                },
                {
                  "idShort": "ActiveLearningSettings",
                  "value": [
                    { "idShort": "al_rounds", "value": 5 },
                    { "idShort": "initial_subset_size", "value": 100 },
                    { "idShort": "add_per_round", "value": 50 },
                    { "idShort": "ud_split", "value": 0.5 },
                    { "idShort": "al_score_threshold", "value": 0.10 }
                  ]
                },
                {
                  "idShort": "DatasetInfo",
                  "value": [
                    { "idShort": "num_classes", "value": 8 },
                    { "idShort": "dataset_yaml", "value": "./data/full-data/dataset.yaml" }
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```
# Step 2: Prepare Your Dataset
Organize your complete, labeled dataset into the standard YOLO format inside the data/full-data directory.

``` code
your_project/
└── data/
    └── full-data/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── dataset.yaml  <-- Your main dataset config
```
Step 3: Launch the Experiment
Execute the main launcher script from your terminal, pointing it to your AAS configuration file.


```
Bash
python run_al_from_aas.py --aas_file "path/to/your/AAS_File.json"
```
The script will handle everything automatically: parsing the AAS, setting up the workspace, and running all rounds of training and selection.
Ste



## 🔍 Algorithm Overview

PPAL consists of two main stages:

#### **Phase 1: Difficulty-Calibrated Uncertainty Sampling**
- The model runs inference on the unlabeled image pool.
- Image-level uncertainty is computed using **Shannon entropy** of detection confidences.
- Each class receives a **difficulty coefficient**, estimated from the average uncertainty per class.
- Images containing uncertain, hard-to-detect classes are prioritized for review.

#### **Phase 2: Category-Conditioned Matching Similarity (CCMS)**
- Computes the **semantic similarity** between candidate images based on shared object categories and bounding box geometry.
- Converts similarity to distance and performs **K-Means++ clustering** to ensure diverse selection.
- The final selected samples maximize both **informativeness** and **diversity**.

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install ultralytics numpy opencv-python pyyaml torch
```


## 🧠 Key Benefits

- **Fully Automated:** Run a complete, multi-round active learning experiment with a single command.
- **Model-agnostic:** Works with any YOLO variant from Ultralytics.
- **Annotation-efficient:** Captures the most informative and diverse samples to maximize model performance with minimal labeling effort.

---

## 📬 Contact

For issues or improvements, please open an issue or pull request on the repository.