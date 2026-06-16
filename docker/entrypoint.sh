#!/bin/sh
# SmartSDG pipeline entrypoint
# Usage: docker run smartsdg <mode> [args...]
#
# Modes:
#   info        Print torch/cuda versions (default)
#   dashboard   Start the web dashboard on :8765
#   train       Run OD training   (pass --aas_file <path> and any YOLO flags)
#   al          Run Active Learning
#   cl          Run Continual Learning
#   inference   Run batch inference
#   shell       Drop into bash (for debugging)

set -e

MODE=${1:-info}
shift || true   # remaining args passed through to the stage script

case "$MODE" in

  info)
    python - <<'PY'
import torch, ultralytics
print("SmartSDG image ready")
print(f"  torch      : {torch.__version__}")
print(f"  ultralytics: {ultralytics.__version__}")
print(f"  CUDA       : {torch.cuda.is_available()} ({torch.version.cuda})")
if torch.cuda.is_available():
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
PY
    ;;

  python)
    # dashboard/server.py calls: docker run ... smartsdg python <script> [args]
    exec python "$@"
    ;;

  dashboard)
    echo "[entrypoint] Starting dashboard on http://0.0.0.0:8765 ..."
    exec python dashboard/server.py "$@"
    ;;

  train)
    echo "[entrypoint] OD Training ..."
    exec python OD_training/main.py --mode train "$@"
    ;;

  al)
    echo "[entrypoint] Active Learning ..."
    exec python AL/main_al.py "$@"
    ;;

  cl)
    echo "[entrypoint] Continual Learning ..."
    exec python CL/main_cl.py "$@"
    ;;

  inference)
    echo "[entrypoint] Inference ..."
    exec python inference/run_inference.py "$@"
    ;;

  shell)
    exec bash
    ;;

  *)
    # Fall-through: treat first arg as a raw python script path
    echo "[entrypoint] Running: python $MODE $*"
    exec python "$MODE" "$@"
    ;;

esac
