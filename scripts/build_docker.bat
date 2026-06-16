@echo off
REM ===================================================
REM  SmartSDG Pipeline v2 — Docker build helper
REM
REM  Builds the smartsdg image covering all Python
REM  stages (train / AL / CL / inference / dashboard).
REM  NOT for Isaac Sim (SDG).
REM
REM  After building, start the dashboard with:
REM    docker compose -f docker/docker-compose.yml up dashboard
REM  Then open http://localhost:8765
REM
REM  Requires Docker Desktop. First build downloads ~5 GB.
REM ===================================================
cd /d "%~dp0.."

echo [build] Building smartsdg image ...
docker build -t smartsdg -f docker/Dockerfile .

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [build] Done.  Run the dashboard with:
    echo   docker compose -f docker/docker-compose.yml up dashboard
    echo.
    echo   Or a one-off training run:
    echo   docker run --gpus all -v "%CD%:/workspace" smartsdg train --aas_file AAS/datasets/...
) else (
    echo [build] FAILED. Check Docker Desktop is running.
)
pause
