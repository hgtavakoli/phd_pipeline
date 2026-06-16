@echo off
REM Build the smartsdg-train image for the Python stages (training/AL/CL).
REM Needs Docker Desktop running. First build downloads ~5 GB and takes a while.
cd /d "%~dp0.."
docker build -t smartsdg-train -f docker/Dockerfile .
pause
