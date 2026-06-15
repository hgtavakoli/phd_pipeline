# Legacy AL configs (not used by the dashboard / AAS flow)

config_yolo.py and evaluate_yolo.py are the old standalone-CLI Active Learning
config and evaluator. The live pipeline drives AL through AAS files + the
dashboard (run_al_from_aas.py -> main_yolo.py), which build their config dicts
directly. Kept for reference; safe to delete.
