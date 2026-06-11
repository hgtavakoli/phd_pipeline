@echo OFF
:: ====================================================================================
:: SmartSDG Master Pipeline Script
::
:: This script orchestrates the entire synthetic data generation process.
:: 1. It takes an AAS file as input.
:: 2. It runs a Python script to convert the AAS model into a YAML config file.
:: 3. It runs the main Isaac Sim script, pointing it to the newly generated config.
:: ====================================================================================

:: Enable a more powerful way of handling variables.
setlocal EnableDelayedExpansion

:: --- 1. Argument Processing ---
:: Check if the first argument (the AAS file path) was provided.
IF "%~1"=="" (
    echo.
    echo [ERROR] You must provide a path to an AAS JSON file as the first argument.
    echo.
    echo Usage: .\run_pipeline.bat [PATH_TO_AAS_FILE] [options_for_main_script]
    exit /b 1
)

:: Store the AAS file path from the first argument (%~1 removes quotes).
set "AAS_INPUT_FILE=%~1"

:: Create a new, empty variable to hold all other arguments for the main script.
set "MAIN_SCRIPT_ARGS="

:: This flag is used to identify and skip the first argument in the loop.
set "skip_first=true"

:: Loop through every argument passed to this batch script.
for %%A in (%*) do (
    if "!skip_first!"=="true" (
        :: This is the first argument (the AAS file), so we just ignore it
        :: and update the flag so we don't skip any others.
        set "skip_first=false"
    ) else (
        :: For every subsequent argument, append it to our new variable.
        set "MAIN_SCRIPT_ARGS=!MAIN_SCRIPT_ARGS! %%A"
    )
)

echo [PIPELINE] Captured AAS Input File: %AAS_INPUT_FILE%
echo [PIPELINE] Arguments for Isaac Sim:!MAIN_SCRIPT_ARGS!
echo.

:: --- 2. Configuration Generation ---
set PYTHON_FOR_CONVERTER=python
set CONFIG_GENERATOR_SCRIPT=./src/aas_config_converter.py

echo [STEP 1] Running AAS to YAML Converter...
echo          Input File: %AAS_INPUT_FILE%
echo.

:: Execute the converter script.
%PYTHON_FOR_CONVERTER% %CONFIG_GENERATOR_SCRIPT% "%AAS_INPUT_FILE%"

:: Check the exit code of the Python script. If it's not 0, an error occurred.
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Configuration generation failed. Aborting pipeline.
    exit /b %ERRORLEVEL%
)
echo.
echo [STEP 1] Finished Successfully.
echo.

:: --- 3. Isaac Sim Data Generation ---
set ISAAC_SIM_PYTHON=../../omniverse/python.bat
set MAIN_SIM_SCRIPT=main.py
set GENERATED_CONFIG_FILE=src\config\config_final.yaml

echo [STEP 2] Starting Isaac Sim Generation Pipeline...
echo          Using generated config: %GENERATED_CONFIG_FILE%
echo.

:: Execute the main Isaac Sim script.
:: THE FIX IS HERE: A space is added between "%GENERATED_CONFIG_FILE%" and !MAIN_SCRIPT_ARGS!
%ISAAC_SIM_PYTHON% %MAIN_SIM_SCRIPT% --config "%GENERATED_CONFIG_FILE%"!MAIN_SCRIPT_ARGS!

echo.
echo [PIPELINE] All tasks finished.
endlocal