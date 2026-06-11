

# SmartSDG: AAS-Driven Synthetic Data Generation for Isaac Sim

<!-- A high-quality banner image or GIF of the output is highly recommended here -->
<!-- ![Banner](docs/banner.gif) -->

**SmartSDG** is an automated pipeline for generating high-quality, randomized synthetic datasets for computer vision tasks using **NVIDIA Omniverse Isaac Sim**.

The defining feature of this framework is its **AAS-Driven Architecture**. Instead of relying on complex manual configuration files, the entire process is controlled by an **Asset Administration Shell (AAS)** model. This "digital twin" of the configuration ensures that the data generation process is robust, reproducible, and seamlessly integrated into Industry 4.0 workflows.

A master script orchestrates the entire pipeline: reading the AAS, generating the necessary runtime configurations, launching the simulation, and finally, converting the output into a ready-to-use format for training AI models.

## Table of Contents

1.  [⭐ **Core Features**](#-core-features)
2.  [🔧 **How the Pipeline Works**](#-how-the-pipeline-works)
3.  [✅ **Prerequisites**](#-prerequisites)
4.  [🚀 **Quick Start Tutorial**](#-quick-start-tutorial)
5.  [⚙️ **Configuration Guide**](#️-configuration-guide)
    -   [The Role of the AAS](#the-role-of-the-aas)
    -   [Verifying Script Paths](#verifying-script-paths)
6.  [💻 **Usage Guide**](#-usage-guide)
    -   [Command-Line Syntax](#command-line-syntax)
    -   [Generation Modes Explained](#generation-modes-explained)
    -   [Detailed Parameter Reference](#detailed-parameter-reference)
7.  [🧩 **Extending the Framework**](#-extending-the-framework)
    -   [How to Add New 3D Models](#how-to-add-new-3d-models)
8.  [📂 **Project Structure**](#-project-structure)
9.  [📜 **License**](#-license)

---

## ⭐ Core Features

-   **AAS-Driven Configuration**: Eliminates manual config files by using an AAS as the single source of truth for all parameters.
-   **Fully Automated Pipeline**: A single command handles everything: AAS parsing, config generation, Isaac Sim execution, and YOLO conversion.
-   **Two Powerful Generation Modes**:
    -   **Single-Object Mode**: Generates datasets of individual components, perfect for classification or instance segmentation.
    -   **Multi-Object Mode**: Creates complex scenes with multiple objects scattered together, ideal for object detection.
-   **Rich Scene Randomization**:
    -   **Pose Randomization**: Objects are placed and rotated within a defined 3D volume.
    -   **Lighting Randomization**: The intensity and color temperature of scene lights are varied to simulate different conditions.
-   **Integrated YOLO Conversion**: Includes an optional, built-in step to automatically convert the generated bounding box data into the widely-used YOLO format for immediate use in training pipelines.

---

## 🔧 How the Pipeline Works

The project is orchestrated by a master script, `run_pipeline.bat`, which executes a three-step process. This automation ensures consistency and removes the risk of manual error.

```bash
graph TD
    A[Start: User runs run_pipeline.bat] --> B{Step 1: AAS to YAML Conversion};
    B -->|Success| C{Step 2: Isaac Sim Generation};
    B -->|Failure| F[Stop & Display Error];
    C -->|Success| D{Step 3 (Optional): YOLO Conversion};
    C -->|Failure| F;
    D --> E[End: Dataset is Ready];

    subgraph "User Provides"
        U1["1. AAS Model File (.json)"];
        U2["2. Generation Arguments (e.g., --num_samples)"];
    end

    subgraph "Orchestrator Executes"
        B -- "Calls aas_config_converter.py" --> P1((Python Script));
        C -- "Calls main.py" --> P2((Isaac Sim Script));
        D -- "Calls yolo_converter.py" --> P3((Python Script));
    end

    U1 --> B;
    U2 --> C;
```

1.  **AAS to YAML Conversion**: The orchestrator first calls a Python script that reads your specified AAS `.json` file. It extracts all parameters and **overwrites** a temporary `config_final.yaml` file. This is the key to ensuring the simulation always uses the most up-to-date configuration from your AAS.
2.  **Isaac Sim Generation**: Next, the orchestrator launches Isaac Sim, pointing it to the `config_final.yaml` created moments before. Isaac Sim then generates the raw RGB images and bounding box data based on your command-line arguments.
3.  **YOLO Conversion (Optional)**: If you provide the `--convert_to_yolo` flag, the pipeline automatically processes the raw output, creating the standard `.txt` label files required for training YOLO models.

---

## ✅ Prerequisites

Before you begin, please ensure your system meets the following requirements:

1.  **NVIDIA Omniverse & Isaac Sim**: A working installation of Isaac Sim (2022.2.1 or later is recommended).
2.  **Anaconda/Miniconda**: A separate, standard Python environment is required for the data processing scripts.
3.  **Required Python Libraries**: These libraries are essential for the AAS conversion step. Install them into your Anaconda environment:
    ```bash
    # Open an Anaconda Prompt and run:
    pip install "aas-core3-smc" PyYAML
    ```

---

## 🚀 Quick Start Tutorial

This tutorial will guide you through generating your first dataset in just a few steps.

**Step 1: Get the Project Files**
Clone this repository to your local machine and ensure you have downloaded all the necessary 3D assets (`.usd` files, textures, etc.).

**Step 2: Place Your Assets**
Create an `assets` folder inside the `SmartSDG` directory and place all your 3D models and textures inside it.

**Step 3: Configure Your AAS Model**
Open your AAS `.json` file (e.g., `AAS/CA_Modul-OD_pipeline_v1.1.json`) in a text editor. This is your central control panel. Find the `SubmodelElementList` with the `idShort` of **`COMPONENT_USD_MAP`** and ensure the filenames listed there match the `.usd` files you placed in your `assets` folder.

**Step 4: Verify Script Paths**
Open `run_pipeline.bat` in a text editor. Check that the `ISAAC_SIM_PYTHON` variable points to the correct location of your Omniverse installation's `python.bat`.

```batch
:: This path must be correct for your system!
set ISAAC_SIM_PYTHON=../../omniverse/python.bat
```

**Step 5: Run the Pipeline!**
Open a terminal (like CMD or Anaconda Prompt), navigate to the `SDG/SmartSDG` directory, and run the following command:

```bash
:: This command generates 10 images of the 'GlueGun' and prepares them for YOLO training.
.\run_pipeline.bat "..\AAS\CA_Modul-OD_pipeline_v1.1.json" --component_class GlueGun --light_mode High --num_samples 10 --load_env plane --convert_to_yolo
```

Once the script finishes, your newly generated dataset will be in the `output` directory, ready to use.

---

## ⚙️ Configuration Guide

### The Role of the AAS
In this framework, the **Asset Administration Shell (AAS)** is the master configuration source. You should not edit the `config_final.yaml` file manually, as it is overwritten every time you run the pipeline.

All persistent changes to your data generation setup should be made in your AAS `.json` file. This includes:
-   **File Paths**: Defining the locations of your assets.
-   **Component Definitions**: Listing all available 3D models and their class names.
-   **Camera Properties**: Setting image resolution, focal length, etc.
-   **Scene Parameters**: Defining the scale and position of randomization volumes.

### Verifying Script Paths
The `run_pipeline.bat` script relies on two key path variables. If you encounter errors, check these first:
-   `PYTHON_FOR_CONVERTER`: Should be set to `python`. If your Anaconda environment is not in your system's PATH, you may need to provide the full path to `python.exe`.
-   `ISAAC_SIM_PYTHON`: Must be the relative or absolute path to the `python.bat` (or `python.sh` on Linux) inside your Isaac Sim installation directory.

---

## 💻 Usage Guide

### Command-Line Syntax
The master script is invoked with the following structure:

```bash
.\run_pipeline.bat "[PATH_TO_AAS_FILE]" [SIMULATION_OPTIONS...]
```

-   The first argument is **always** the path to the AAS model, enclosed in quotes.
-   All subsequent arguments are flags that control the behavior of the Isaac Sim generation process.

### Generation Modes Explained

You can generate data in two distinct modes, controlled by command-line flags.

####  Mode 1: Single-Object (`--component_class`)
This mode is for generating images of one specific object at a time. It is activated by using the `--component_class` flag. The simulation will randomly place and orient only the specified object in the scene for each image. This is ideal for creating datasets for **image classification** or **instance segmentation**.

```bash
# Generates a dataset focusing only on the 'BlackCabin' object.
.\run_pipeline.bat "..\AAS\..." --component_class BlackCabin --num_samples 100
```

#### Mode 2: Multi-Object (`--multi_object`)
This mode is for generating complex scenes with many different objects interacting. It is activated by using the `--multi_object` flag. In each image, the simulation will take *all* components defined in your AAS and scatter them randomly together in the scene. This is perfect for creating datasets for **object detection**.

```bash
# Generates a dataset where all objects appear together in random arrangements.
.\run_pipeline.bat "..\AAS\..." --multi_object --num_samples 200
```

### Detailed Parameter Reference

This table provides a comprehensive reference for all available command-line arguments.

| Argument | Required? | Description & Values |
| :--- | :--- | :--- |
| **`[AAS_FILE_PATH]`** | **Yes** | The path to the AAS JSON model. This is the **first** argument, not preceded by a flag. |
| **`--light_mode`** | **Yes** | Sets the lighting preset. **Values**: `Low`, `Medium`, `High`. |
| `--component_class` | If not `--multi_object` | Specifies which single object to render. The name must match a key in the AAS `COMPONENT_USD_MAP`. |
| `--multi_object` | No | A flag to enable multi-object mode. If present, `--component_class` is ignored. |
| `--num_samples` | No (Default: 10) | The total number of images to generate. |
| `--load_env` | No (Default: `false`) | Controls the background. **Values**: `true` (full 3D scene), `plane`, `false` (void background). |
| `--convert_to_yolo`| No | A flag to enable automatic conversion of the output to YOLO format. |
| `--filter_overlaps`| No | A flag to remove images where bounding boxes overlap too much. Requires `--convert_to_yolo`. |
| `--seed` | No | An integer used to seed the random number generators. Using the same seed guarantees identical "random" results, which is useful for debugging and reproducibility. |

---

## 🧩 Extending the Framework

### How to Add New 3D Models

Adding a new component to the data generation pool is a simple, two-step process focused on the AAS model.

1.  **Place the Asset File**: Copy your new component's `.usd` file into your project's `assets` directory.

2.  **Update the AAS Model**: Open your AAS `.json` file and find the `SubmodelElementList` with the `idShort` of **`COMPONENT_USD_MAP`**. Add a new element to its `value` array that defines your new component. You must match the structure of the existing entries.

    *Example*: To add a file named `NewSensor.usd` and give it the class name `Sensor`:
    ```json
    {
      "idShort": "Sensor",
      "modelType": "Property",
      "valueType": "string",
      "value": "NewSensor.usd"
    }
    ```
    *(Note: The exact AAS structure for a Property or Reference may vary slightly. Always match the format of the other components in the list).*

That's it! You can now run the pipeline using `--component_class Sensor`, and it will be included in `--multi_object` runs.

---

## 📂 Project Structure

A brief overview of the key files and directories in the project.

```
/
+-- AAS/                          # Contains the master AAS model(s).
|   +-- CA_Modul-OD_pipeline_v1.1.json
|
+-- omniverse/                    # Should contain your Isaac Sim installation.
|   +-- python.bat
|
+-- SDG/
    +-- SmartSDG/
        +-- run_pipeline.bat      # The master script to run the entire pipeline.
        +-- main.py               # The main entry point for the Isaac Sim application.
        +-- src/
        |   +-- aas_config_converter.py # Python script that converts the AAS model to YAML.
        |   +-- config_process.py     # Loads and validates the YAML config.
        |   +-- generation.py         # The core Replicator logic for Isaac Sim.
        |   +-- yolo_converter.py     # Handles the conversion to YOLO format.
        |   +-- config/
        |       +-- config_final.yaml # Temporary file. Auto-generated and overwritten on every run.
        +-- assets/                   # Your 3D models (.usd) and textures should be placed here.
        +-- output/                   # Default directory where generated datasets are saved.
```

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/licenses/MIT) file for details.