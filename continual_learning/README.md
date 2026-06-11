
# Class-Incremental Object Detection with YOLO

This project provides a pipeline for Class-Incremental Learning (CIL) for object detection using the Ultralytics YOLO framework. The primary goal is to train a model on new classes sequentially without drastically forgetting previously learned classes—a common problem known as "catastrophic forgetting."

The core of this approach is a **replay-based strategy** using a custom **Balanced Batch Sampler**. This ensures that during each training step, the model sees a mix of new data (for learning new classes) and a smaller, representative set of old data (for retaining knowledge).

## Features

-   **AAS Driven:** The entire workflow is controlled by a structured AAS (Asset Administration Shell) JSON file.
-   **Incremental Training:** Add new object classes to an existing YOLO model in distinct steps.
-   **Balanced Data Loading:** Utilizes a custom PyTorch `Sampler` to create batches with a specific, user-defined ratio of "new" vs. "replay" data.
-   **Robust and Informative:** The sampler handles edge cases (like small replay buffers) and provides a detailed summary of the batching strategy before each run.
-   **Reliable Parsing:** Uses the `aas-core3-sdk` for robust parsing of the AAS configuration file.

## Project Structure

```
continual_learning/
├── cl_src/
│   ├── aas_cl_config_converter.py  # Robustly parses the AAS JSON file.
│   ├── cl_orchestrator.py          # Main logic for an incremental step.
│   └── data_utils.py               # Data prep functions and the BalancedBatchSampler.
├── main_cl.py                      # Core training script (called by the wrapper).
├── run_from_aas.py                 # The main entry point for launching the pipeline.
└── README.md                       # This file.
```

-   **`run_from_aas.py`**: **This is the main script you will run.** It reads the AAS JSON, extracts parameters, and executes the training pipeline.
-   **`cl_src/aas_cl_config_converter.py`**: A dedicated module that uses the `aas-core3-sdk` to reliably navigate the complex AAS structure and extract the necessary parameters for continual learning.
-   **`main_cl.py`**: The underlying script that orchestrates a single training step. It is called automatically by `run_from_aas.py`.

## Setup and Installation

1.  **Prerequisites**:
    -   Python 3.10+
    -   PyTorch
    -   NVIDIA GPU with CUDA

2.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd continual_learning
    ```

3.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you don't have a `requirements.txt`, you'll need `pip install ultralytics torch pyyaml aas-core3-sdk`)*

## How to Run the Pipeline (AAS Workflow)

The entire process is driven by a single AAS JSON configuration file and the main wrapper script.

### Step 1: Configure the AAS JSON File

Before running, you must configure your AAS JSON file (`CA_Modul-OD_pipeline_v0.7.json` or similar). The wrapper script will look for the following path and properties:
**`CA_Modul -> OD_Module -> ODOptimisation -> ContinualLearning`**

Ensure the `ContinualLearning` property list contains the following `idShort` entries with your desired values:

```json
{
  "idShort": "ContinualLeaning", // Note: The parser handles the typo "Leaning"
  "value": [
    {
      "idShort": "input_model",
      "value": "\"C:/path/to/your/previous/model/best.pt\""
    },
    {
      "idShort": "current_classes",
      "value": "\"class_A,class_B,class_C\""
    },
    {
      "idShort": "new_classes",
      "value": "\"class_D,class_E\""
    },
    {
      "idShort": "replay_data_dir",
      "value": "\"C:/path/to/your/replay_data\""
    },
    {
      "idShort": "new_data_dir",
      "value": "\"C:/path/to/your/new_data\""
    },
    {
      "idShort": "exp_name",
      "value": "step2_learn_D_E"
    },
    {
      "idShort": "replay_ratio",
      "value": "0.4"
    },
    // --- Optional Parameters ---
    {
      "idShort": "val_replay_data_dir",
      "value": "\"C:/path/to/your/val_replay_data\""
    },
    {
      "idShort": "val_new_data_dir",
      "value": "\"C:/path/to/your/val_new_data\""
    },
    {
      "idShort": "epochs",
      "value": "100"
    },
    {
      "idShort": "batch_size",
      "value": "16"
    }
  ]
}
```

### Step 2: Execute the Wrapper Script

Once your JSON is configured, run the `run_from_aas.py` script, passing the path to your AAS file as the only argument.

```bash
python run_from_aas.py "C:/Users/hota02/Documents/SDG/AAS/CA_Modul-OD_pipeline_v0.7.json"
```

The script will handle everything else: parsing the AAS, constructing the necessary command, and launching the training.

## Interpreting the Console Output

The output is now separated into two stages:

1.  **AAS Parsing Stage:** The wrapper script first confirms it has successfully parsed your file.
    ```
    --- Initializing AAS Converter ---
    📄 Loading AAS configuration from: C:\...\CA_Modul-OD_pipeline_v0.7.json
    --> Navigating AAS path: OD_Module -> ODOptimisation -> ContinualLeaning
    ✅ Successfully extracted raw parameters from AAS.
    ✅ Parameters cleaned and validated.
    ```
2.  **Command Execution:** The wrapper will print the final command it's about to run for verification.
    ```
    =========================== Executing Command ===========================
    "python" "main_cl.py" --current_classes "..." --new_classes "..." ...
    ===========================================================================
    ```
3.  **Training Stage:** You will then see the familiar output from the training script, including the **Batch Sampler Info**, YOLO model summary, and training progress.
    ```
    ========================= Batch Sampler Info =========================
      - Total unique NEW images found:    ...
      - Replay Samples per Batch:         ...
    ========================================================================

    🚀 Initializing model for training...
    - Callback triggered: Replacing default train_loader with custom balanced loader.
    ...
    (YOLO training progress bar)
    ```

## How to See and Evaluate the Results

The evaluation process remains the same.

1.  **Get the Final Model:** Your new model is located in the output directory specified in your AAS config (e.g., `results/cl_runs/step2_learn_D_E/weights/best.pt`).
2.  **Prepare a Combined Test Set:** Create a validation dataset containing images for **all** classes learned so far.
3.  **Run Standard YOLO Validation:**
    ```bash
    yolo val model=/path/to/new/best.pt data=combined_test_set.yaml
    ```
4.  **Analyze the Results:** Check the per-class mAP scores. Success is when the scores for both old and new classes are high. If old class scores have dropped significantly, consider increasing the `replay_ratio` in the AAS file for the next step.

## Next Incremental Step

To continue learning, simply:
1.  **Update the AAS JSON file:**
    -   Change `input_model` to the path of the model you just trained.
    -   Update `current_classes` to include the classes you just learned.
    -   Set `new_classes` to the next set of classes.
    -   Update paths for the new `replay_data_dir` and `new_data_dir`.
2.  **Re-run the `run_from_aas.py` script.**

<br>

<details>
<summary><b>Advanced: Manual Execution (main_cl.py)</b></summary>

For debugging or direct control, you can bypass the AAS wrapper and run `main_cl.py` directly. This requires you to build the full command with all arguments manually.

### Anatomy of the Command
```bash
python main_cl.py \
    --current_classes "class_A,class_B" \
    --new_classes "class_C,class_D" \
    --model_weights "/path/to/previous/best.pt" \
    --replay_data_dir "/path/to/replay_data_for_A_B" \
    --new_data_dir "/path/to/new_data_for_C_D" \
    --exp_name "manual_run" \
    --replay_ratio 0.4
```
### Argument Explanations
-   `--current_classes`: Comma-separated string of known class names.
-   `--new_classes`: Comma-separated string of new class names.
-   `--model_weights`: Path to the input `best.pt` file.
-   `--replay_data_dir`: Path to the replay (old) training data.
-   `--new_data_dir`: Path to the new training data.
-   `--val_replay_data_dir` (Optional): Path to the validation set for old classes.
-   `--val_new_data_dir` (Optional): Path to the validation set for new classes.
-   `--exp_name`: Name for the output directory.
-   `--batch_size`: Total batch size.
-   `--epochs`: Number of training epochs.
-   `--replay_ratio`: The proportion (0.0 to 1.0) of each batch dedicated to replay data.

</details>