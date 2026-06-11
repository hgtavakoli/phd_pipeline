@echo off
REM ===================================================
REM  SmartSDG Pipeline v2 — Dashboard launcher
REM  Starts the local dashboard server and opens the
REM  browser at http://127.0.0.1:8765
REM
REM  TIP: activate your training Python env first
REM  (e.g. `conda activate yolov12`) so the Training
REM  button uses the right interpreter.
REM ===================================================
cd /d "%~dp0.."
python dashboard\server.py
pause
