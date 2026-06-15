"""
Portable machine-path resolution for the SmartSDG pipeline.

Three machine-specific paths are needed at runtime:

    isaac_python   Isaac Sim's bundled Python launcher (python.bat) for SDG
    train_python   Python interpreter with torch/ultralytics/aas_core3 for
                   OD / AL / CL / inference
    val_real       folder of real validation images (sim-to-real evaluation)

Resolution order for each (first hit wins):
    1. environment variable        SMARTSDG_ISAAC_PYTHON / _TRAIN_PYTHON / _VAL_REAL
    2. pipeline_config.json         at the project root (copy from the .example)
    3. auto-detection               common install locations on this machine
    4. sensible fallback            ('python' on PATH for train; None otherwise)

This means the pipeline runs on a new machine by either setting three env vars
or editing one JSON file — no source edits. `python config_paths.py` prints the
resolved paths and whether each exists, for quick diagnosis.
"""

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "pipeline_config.json"


def _load_json():
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8-sig"))
        except Exception as e:
            print(f"[config_paths] WARNING: could not parse {CONFIG_FILE.name}: {e}", file=sys.stderr)
    return {}


_CFG = _load_json()


def _first_existing(candidates):
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None


def resolve_isaac_python():
    """Isaac Sim python.bat (SDG)."""
    env = os.environ.get("SMARTSDG_ISAAC_PYTHON")
    if env:
        return env
    if _CFG.get("isaac_python"):
        return _CFG["isaac_python"]
    home = Path.home()
    return _first_existing([
        home / "Documents" / "omniverse" / "python.bat",
        Path(r"C:\Users\hota02\Documents\omniverse\python.bat"),
        Path(r"C:\isaac-sim\python.bat"),
        Path("/isaac-sim/python.sh"),
    ]) or str(home / "Documents" / "omniverse" / "python.bat")


def resolve_train_python():
    """Python interpreter for OD/AL/CL/inference (needs torch+ultralytics+aas_core3)."""
    env = os.environ.get("SMARTSDG_TRAIN_PYTHON")
    if env:
        return env
    if _CFG.get("train_python"):
        return _CFG["train_python"]
    home = Path.home()
    # common conda env locations, then the interpreter running this process,
    # then plain 'python' on PATH
    found = _first_existing([
        home / "AppData" / "Local" / "anaconda3" / "envs" / "yolov12" / "python.exe",
        home / "anaconda3" / "envs" / "yolov12" / "python.exe",
        home / "miniconda3" / "envs" / "yolov12" / "python.exe",
    ])
    if found:
        return found
    # if the current interpreter has ultralytics, use it
    try:
        import importlib.util
        if importlib.util.find_spec("ultralytics") is not None:
            return sys.executable
    except Exception:
        pass
    return shutil.which("python") or sys.executable


def resolve_val_real():
    """Folder of real validation images (sim-to-real eval). May be None."""
    env = os.environ.get("SMARTSDG_VAL_REAL")
    if env:
        return env
    if _CFG.get("val_real"):
        return _CFG["val_real"]
    home = Path.home()
    return _first_existing([
        home / "Documents" / "sd_light_od" / "cluster_experiments" / "data" / "val_real",
        ROOT / "data" / "val_real",
    ])  # may be None — AL still runs, just without real-image validation


if __name__ == "__main__":
    for name, val in [("isaac_python", resolve_isaac_python()),
                      ("train_python", resolve_train_python()),
                      ("val_real", resolve_val_real())]:
        exists = "OK" if (val and Path(val).exists()) else "MISSING"
        print(f"  {name:14s} = {val}   [{exists}]")
