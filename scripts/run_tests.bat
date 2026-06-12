@echo off
REM ===================================================
REM  SmartSDG Pipeline v2 — self-test
REM
REM  run_tests.bat          quick static checks (~10 s)
REM  run_tests.bat full     + real SDG generation and
REM                         training smoke test (~5-10 min)
REM ===================================================
cd /d "%~dp0.."
SET PY=C:\Users\hota02\AppData\Local\anaconda3\envs\yolov12\python.exe

IF /I "%1"=="full" (
    "%PY%" -X utf8 scripts\test_pipeline.py --full
) ELSE (
    "%PY%" -X utf8 scripts\test_pipeline.py
)
pause
