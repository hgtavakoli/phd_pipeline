#!/bin/bash

# Stop the script if any command fails
set -e

echo "--- Step 1: Generating configuration from AAS file... ---"

# --- CONFIGURATION ---
# Use a standard Python 3 for the conversion script
PYTHON_FOR_CONVERTER="python3" 

# Path to the AAS converter script
CONFIG_GENERATOR_SCRIPT="aas_config_converter.py" # Or cl_src/config_generator.py if you moved it

# --- EXECUTION ---
# Run the config generator.
# This will create the 'config/config_final.yaml' file.
$PYTHON_FOR_CONVERTER $CONFIG_GENERATOR_SCRIPT

# Check if the generator script was successful before proceeding
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Configuration generation failed. Aborting pipeline."
    exit 1
fi

echo "✅ Configuration generated successfully."
echo ""
echo "--- Step 2: Starting Isaac Sim simulation pipeline... ---"

# --- CONFIGURATION ---
# Path to Isaac Sim's python executable
ISAAC_SIM_PYTHON="../../omniverse/python.sh" # Adjust this path to your Isaac Sim python.sh

# Path to the main simulation script
MAIN_SIM_SCRIPT="main.py"

# Path to the generated config file that the main script will use
GENERATED_CONFIG_FILE="config/config_final.yaml"

# --- EXECUTION ---
# Run the main Isaac Sim script, passing the generated config file path
# and all other command-line arguments (`$@`) given to this shell script.
$ISAAC_SIM_PYTHON $MAIN_SIM_SCRIPT --config $GENERATED_CONFIG_FILE "$@"

echo ""
echo "--- Pipeline finished. ---"