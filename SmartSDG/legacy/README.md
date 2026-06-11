# Legacy — not part of the live pipeline

Files moved here during the v2 cleanup (2026-06-11). Nothing in this folder is
imported or executed by `main.py`. Kept for reference only; safe to delete.

| File | What it was |
|---|---|
| `SmartSDG.py` | Old monolithic generation script, superseded by `main.py` + `src/` |
| `temp_main.py` | Scratch/test entry point |
| `worker_writer.py`, `yolo_writer.py` | Early writer experiments, superseded by `src/yolo_converter.py` |
| `cl_src_root/`, `cl_src_src/` | Stale copies of `config_final.yaml` from the old `cl_src` layout |
