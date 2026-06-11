# SmartSDG: End-to-End Synthetic Data Generation Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Built with](https://img.shields.io/badge/Built_with-NVIDIA_Isaac_Sim-76B900.svg)](https://developer.nvidia.com/isaac-sim)

## Reference Arcitecture

![RA](Documents/RA_thesis.drawio.svg)

> **Quick Sample Run:**
> ```bash
> ./python.sh main.py --component_class GlueGun --light_mode M --num_samples 100 --load_env plane --convert_to_yolo --config cl_src/config/config.yaml
> ```

**SmartSDG** is a powerful and modular framework for generating synthetic datasets for object detection tasks using **NVIDIA Omniverse Isaac Sim**. It is designed to be highly configurable through an external YAML file, allowing users to control every aspect of the simulation—from asset paths to complex randomization logic.

The framework automates the entire pipeline, from generating raw image data with varied, randomized scenes to converting annotations into the popular **YOLO format** and optionally filtering the dataset to improve quality.

## ✨ Table of Contents
1.  [⭐ Key Features](#-key-features)
2.  [📂 Project Structure](#-project-structure)
3.  [⚙️ Prerequisites](#️-prerequisites)
4.  [🚀 Installation and Setup](#-installation-and-setup)
5.  [🔧 Configuration (`config.yaml`)](#-configuration-configyaml)
6.  [▶️ Usage](#️-usage)
7.  [💡 How It Works: Technical Overview](#-how-it-works-technical-overview)
8.  [🧩 Extending the Framework](#-extending-the-framework)
9.  [📜 License](#-license)

## ⭐ Key Features

-   **YAML-Driven Configuration**: Control all paths, objects, camera settings, and randomization parameters from a single `config.yaml` file without modifying code.
-   **Portable Path Management**: The script dynamically resolves relative paths, so you don't need to hardcode absolute paths in your configuration.
-   **Dual Generation Modes**:
    -   **Single-Object Mode**: Renders scenes with one component, ideal for classification datasets.
    -   **Multi-Object Mode**: Renders complex scenes with multiple components scattered together for robust object detection.
-   **Advanced Randomization**:
    -   **Object Pose**: Randomizes the position and rotation of components within defined volumes.
    -   **Lighting**: Randomizes the intensity and temperature of scene lights to simulate varied conditions.
-   **Dual Camera Setup**: Captures both a consistent, top-down view and a dynamic, random orbital view in a single run.
-   **End-to-End YOLO Pipeline**:
    -   **Automatic Conversion**: Includes a powerful script to automatically convert Isaac Sim's output into YOLO-compatible image and label files.
    -   **Overlap Filtering**: Optionally filters the final dataset to remove images with significantly overlapping bounding boxes.

## 📂 Project Structure

The framework is organized into logical modules for clarity and maintainability.

````

/SmartSDG/
|
\+-- main.py                     \# Main entry point: handles arguments, config, and orchestrates the pipeline.
|
\+-- src/
|   +-- generation.py           \# The "engine": contains the core Isaac Sim Replicator logic.
|   +-- yolo\_converter.py       \# Handles post-processing: converts to YOLO format and filters overlaps.
|   +-- utils.py                \# Contains reusable helper functions (e.g., camera positioning).
|   +-- config/
|       +-- config.yaml         \# The "control panel": all user-configurable parameters are defined here.
|
\+-- assets/                     \# (User-created) Directory for all 3D models and environments.
|
\+-- Datasets/                   \# (User-created) Base directory where all generated output will be saved.
|
\+-- README.md                   \# This file.

````

## ⚙️ Prerequisites

-   An **NVIDIA GPU** (RTX 20-series or later recommended).
-   A **Linux** host machine.
-   The recommended NVIDIA Driver version **535.129.03** or later.
-   **Docker Engine** and **NVIDIA Container Toolkit**.

## 🚀 Installation and Setup

This is a detailed, step-by-step guide to prepare your host system and launch the container.

### Step 1: Prepare Host Machine

First, ensure your Linux host has the correct NVIDIA drivers, Docker, and the NVIDIA Container Toolkit installed.

1.  **Install NVIDIA Drivers**
    The recommended driver for Isaac Sim is `535.129.03`.
    ```bash
    wget [https://us.download.nvidia.com/XFree86/Linux-x86_64/535.129.03/NVIDIA-Linux-x86_64-535.129.03.run](https://us.download.nvidia.com/XFree86/Linux-x86_64/535.129.03/NVIDIA-Linux-x86_64-535.129.03.run)
    chmod +x NVIDIA-Linux-x86_64-535.129.03.run
    sudo ./NVIDIA-Linux-x86_64-535.129.03.run
    ```

2.  **Install Docker Engine**
    ```bash
    curl -fsSL [https://get.docker.com](https://get.docker.com) -o get-docker.sh
    sudo sh get-docker.sh
    # Add your user to the docker group
    sudo groupadd docker
    sudo usermod -aG docker $USER
    newgrp docker
    # Verify installation
    docker run hello-world
    ```

3.  **Install NVIDIA Container Toolkit**
    ```bash
    # Configure the repository
    curl -fsSL [https://nvidia.github.io/libnvidia-container/gpgkey](https://nvidia.github.io/libnvidia-container/gpgkey) | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L [https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list](https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list) | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    # Install the toolkit and restart docker
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    
    # Verify the toolkit can see your GPU
    docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
    ```

### Step 2: Clone the Project Repository
```bash
git clone <your-repository-url>
cd SmartSDG
````

### Step 3: Asset and Directory Setup

Create the `assets` and `Datasets` folders inside your project root. Place your `.usd` 3D models into the appropriate subdirectories within `assets` as defined in your `config.yaml`.

```bash
mkdir -p assets
mkdir -p Datasets
# Now, copy your .usd files into the assets folder
```

### Step 4: Run the Isaac Sim Container

Execute the following command from your `SmartSDG` project root. This pulls the official Isaac Sim `4.5.0` container and runs it with persistent caching and your project files mounted.

```bash
# Create directories on the host for persistent caching
mkdir -p ~/docker/isaac-sim/cache/kit
mkdir -p ~/docker/isaac-sim/cache/ov
mkdir -p ~/docker/isaac-sim/cache/pip
mkdir -p ~/docker/isaac-sim/cache/glcache
mkdir -p ~/docker/isaac-sim/cache/computecache
mkdir -p ~/docker/isaac-sim/logs
mkdir -p ~/docker/isaac-sim/data
mkdir -p ~/docker/isaac-sim/documents

# Pull and run the container
docker run --name isaac-sim --entrypoint bash -it --runtime=nvidia --gpus all --rm --network=host \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    \
    # Mount project files
    -v "$(pwd)":/workspace/SmartSDG:rw \
    \
    # Mount directories for caching to speed up future runs
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    \
    nvcr.io/nvidia/isaac-sim:4.5.0
```

*Note: By using the `-e "ACCEPT_EULA=Y"` flag, you accept the NVIDIA Omniverse License Agreement*.

### Step 5: Install Python Dependencies Inside Container

Once inside the container's interactive shell, navigate to the project directory and install the required Python libraries.

```bash
cd /workspace/SmartSDG
./python.sh -m pip install PyYAML numpy pillow
```

## 🔧 Configuration (`config.yaml`)

The file `src/config/config.yaml` is the central control panel.

  - **`BASE_PATH` & `OUTPUT_DIR_BASE`**: These are relative paths to your assets and output directories. The defaults should work if you follow the setup guide.
  - **`COMPONENT_USD_MAP`**: Maps class names (e.g., `Cabin`) to their `.usd` filenames. This is where you add new objects.
  - **`YOLO_CONVERSION.CLASS_NAMES`**: **Crucially, this list must exactly match the order and names of the keys in `COMPONENT_USD_MAP`**. This defines the class index for the YOLO labels.
  - **`YOLO_FILTERING`**: Settings for the overlap filter, including an `enabled` flag and the `IOU_THRESHOLD`.

## ▶️ Usage

All commands must be executed from **inside the Docker container** and from the project root directory (`/workspace/SmartSDG`).

### Command-Line Argument Reference

#### `--component_class`

> **Purpose**: Specifies which object to render in **single-object mode**.  
> **Required?**: Yes, for single-object mode. Ignored if `--multi_object` is present.  
> **Values**: Any key from `COMPONENT_USD_MAP` in `config.yaml` (e.g., `Cabin`, `YTrailer`).  
> **Example**: `./python.sh main.py --component_class FChassis --light_mode S`

#### `--light_mode`

> **Purpose**: Sets the lighting preset for the scene.  
> **Required?**: **Yes, this is always required.** \> **Values**: `S`, `M`, or `L`, corresponding to keys in `LIGHT_INTENSITY_MAP`.  
> **Example**: `./python.sh main.py --component_class GlueGun --light_mode L`

#### `--num_samples`

> **Purpose**: Determines the total number of images to generate.  
> **Required?**: No. **Defaults to `10`**.  
> **Values**: Any positive integer.  
> **Example**: `./python.sh main.py --component_class Cabin --light_mode M --num_samples 500`

#### `--load_env`

> **Purpose**: Controls which background/environment is loaded.  
> **Required?**: No. **Defaults to `false`** (void background).  
> **Values**: `true` (full 3D scene), `plane` (simple flat plane), `false` (no environment).  
> **Example**: `./python.sh main.py --component_class YTrailer --light_mode M --load_env plane`

#### `--multi_object`

> **Purpose**: A flag to switch to **multi-object mode**, scattering all components randomly.  
> **Required?**: No.  
> **Note**: When this flag is present, `--component_class` is ignored.  
> **Example**: `./python.sh main.py --multi_object --light_mode S --num_samples 50`

#### `--seed`

> **Purpose**: Seeds the random number generators for reproducible results.  
> **Required?**: No.  
> **Values**: Any integer.  
> **Example**: `./python.sh main.py --component_class Cabin --light_mode M --seed 42`

#### `--config`

> **Purpose**: Specifies a different YAML configuration file to use.  
> **Required?**: No. **Defaults to `src/config/config.yaml`**.  
> **Values**: A file path to a valid YAML config file.  
> **Example**: `./python.sh main.py --config src/config/low_res_test.yaml --multi_object --light_mode M`

## 💡 How It Works: Technical Overview

1.  **Initialization (`main.py`)**: The script parses arguments, loads `config.yaml`, resolves paths, and launches the Isaac Sim `SimulationApp`.
2.  **Replicator Graph (`generation.py`)**: Defines a **Replicator graph**—a sequence of operations to be executed for each frame.
3.  **Randomizers**: Functions that randomize the scene (object pose, lighting) are registered with Replicator and executed by a frame-based trigger.
4.  **Data Writing**: `RenderProduct` objects attached to a `Writer` render the scene from each camera and save the output (RGB images, bounding boxes) to disk.
5.  **Post-Processing (`yolo_converter.py`)**: If enabled, this script reads the raw Isaac Sim output and converts it into a clean, YOLO-formatted dataset.
6.  **Execution**: The `rep.orchestrator.run()` command executes the entire graph until all samples are complete.

## 🧩 Extending the Framework

### Adding New Components

1.  **Place the Asset**: Add your new `.usd` file (e.g., `NewPart.usd`) to the components directory specified in your config.
2.  **Update Config**: Open `src/config/config.yaml` and make two edits:
      - Add a new key-value pair to `COMPONENT_USD_MAP`:
        ```yaml
        COMPONENT_USD_MAP:
          # ... other components
          NewPart: "NewPart.usd"  # <-- Add this line
        ```
      - Add the new class name to `YOLO_CONVERSION.CLASS_NAMES` in the correct order:
        ```yaml
        CLASS_NAMES:
          # ... other class names
          - 'NewPart'  # <-- Add this line
        ```
3.  **Run**: You can now generate data for your new part using `--component_class NewPart`.

## 📜 License

This project is licensed under the MIT License.

```

Copyright (c) [2025] [Hooman Tavakoli Ghinani]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```
