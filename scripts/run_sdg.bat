@echo off
setlocal EnableDelayedExpansion
REM ===================================================
REM  SmartSDG -- Synthetic Data Generation (interactive)
REM
REM  Just double-click and answer the questions, or pass
REM  everything on the command line to skip the prompts:
REM
REM  run_sdg.bat [mode] [component] [light] [samples] [env] [width] [height]
REM    mode      : single | multi
REM    component : GlueGun | Cabin | BlackCabin | YTrailer |
REM                BLid | BTrailer | FChassis | Rchassis | Ylid
REM                (ignored in multi mode -- pass - as placeholder)
REM    light     : Low | Medium | High
REM    samples   : integer
REM    env       : plane | true | false
REM    width     : image width in px   (e.g. 1280)
REM    height    : image height in px  (e.g. 800)
REM
REM  Isaac Sim path is read from pipeline_config.json first (set via the
REM  dashboard SDG tab > Advanced, or add "isaac_python": "C:\\...\\python.bat"
REM  to pipeline_config.json manually). Auto-detection covers common installs.
REM ===================================================

SET ROOT=%~dp0..
SET SMARTSDG=%ROOT%\SmartSDG
SET CONFIG=%SMARTSDG%\src\config\config.yaml

REM --- Resolve Isaac Sim python.bat via config_paths.py -----------------------
REM  Order: pipeline_config.json  ->  env var  ->  auto-detect  ->  hardcoded fallback
SET ISAAC_PYTHON=
python -c "import sys; sys.path.insert(0, r'%ROOT%'); from config_paths import resolve_isaac_python; print(resolve_isaac_python())" > "%TEMP%\_sdg_ipath.txt" 2>nul
if not errorlevel 1 set /p ISAAC_PYTHON=< "%TEMP%\_sdg_ipath.txt"
del "%TEMP%\_sdg_ipath.txt" 2>nul
if "!ISAAC_PYTHON!"=="" (
    set ISAAC_PYTHON=C:\Users\hota02\Documents\omniverse\python.bat
    echo [INFO] config_paths.py not available; using built-in default path.
)
if not exist "!ISAAC_PYTHON!" (
    echo.
    echo  ERROR: Isaac Sim python.bat not found at:
    echo    !ISAAC_PYTHON!
    echo.
    echo  Fix options:
    echo    1. Open the dashboard, go to SDG tab ^> Advanced, set the path and Save.
    echo    2. Add  "isaac_python": "C:\\...\\python.bat"  to pipeline_config.json.
    echo    3. Set env var  SMARTSDG_ISAAC_PYTHON=C:\...\python.bat
    echo.
    pause
    exit /b 1
)

REM --- Parse positional args (skip interactive prompts when all supplied) -----
SET MODE=%1
SET COMPONENT=%2
SET LIGHT=%3
SET SAMPLES=%4
SET ENV=%5
SET WIDTH=%6
SET HEIGHT=%7

IF "!MODE!"=="" (
    echo.
    echo Generation mode:
    echo   1 = SINGLE object  ^(one component per scene^)
    echo   2 = MULTI object   ^(all components in every scene^)
    set /p MODE="Choose mode (1/2 or single/multi) [1]: "
)
IF "!MODE!"=="" SET MODE=single
IF "!MODE!"=="1" SET MODE=single
IF "!MODE!"=="2" SET MODE=multi

IF /I "!MODE!"=="single" (
    IF "!COMPONENT!"=="" (
        echo.
        echo Components: Ylid YTrailer Cabin GlueGun BLid BTrailer FChassis Rchassis BlackCabin
        set /p COMPONENT="Component [GlueGun]: "
    )
)
IF "!COMPONENT!"=="" SET COMPONENT=GlueGun
IF "!COMPONENT!"=="-"  SET COMPONENT=GlueGun

IF "!LIGHT!"=="" set /p LIGHT="Light mode (Low/Medium/High) [Medium]: "
IF "!LIGHT!"=="" SET LIGHT=Medium

IF "!SAMPLES!"=="" set /p SAMPLES="Number of images [100]: "
IF "!SAMPLES!"=="" SET SAMPLES=100

IF "!ENV!"=="" set /p ENV="Environment (plane/true/false) [plane]: "
IF "!ENV!"=="" SET ENV=plane

IF "!WIDTH!"=="" (
    echo.
    echo Resolution presets: 128x80 ^(fast^), 640x480 ^(recommended^), 1280x800, 1920x1080
    set /p WIDTH="Image width  [640]: "
)
IF "!WIDTH!"==""  SET WIDTH=640
IF "!HEIGHT!"=="" set /p HEIGHT="Image height [480]: "
IF "!HEIGHT!"=="" SET HEIGHT=480

SET MODEFLAG=
IF /I "!MODE!"=="multi" (
    SET MODEFLAG=--multi_object --spawn_radius 0.2 --min_separation 0.05
) ELSE (
    SET MODEFLAG=--component_class !COMPONENT!
)

REM Validated quality filters (CLAUDE.md defaults: visibility, truncation, geometric collapse)
SET FILTERS=--min_visibility 0.95 --drop_truncated --filter_overlaps --iou_threshold 0.7 --containment_threshold 0.75

echo.
echo ========================================
echo  SmartSDG -- Synthetic Data Generation
echo ========================================
echo  Isaac Sim  : !ISAAC_PYTHON!
echo  Mode       : !MODE!
IF /I "!MODE!"=="single" echo  Component  : !COMPONENT!
echo  Light mode : !LIGHT!
echo  Samples    : !SAMPLES!
echo  Environment: !ENV!
echo  Resolution : !WIDTH! x !HEIGHT!
echo  Config     : !CONFIG!
echo ========================================
echo.

cd /d "!SMARTSDG!"

call "!ISAAC_PYTHON!" main.py ^
  --config "!CONFIG!" ^
  !MODEFLAG! ^
  --light_mode !LIGHT! ^
  --num_samples !SAMPLES! ^
  --load_env !ENV! ^
  --width !WIDTH! ^
  --height !HEIGHT! ^
  --convert_to_yolo ^
  !FILTERS!

echo.
echo Done. Output at: !SMARTSDG!\Datasets\yolo_experiments\
pause
