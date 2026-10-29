# cl_src/main.py (Corrected Version)

import argparse
import yaml
import os
import re
import sys
from pathlib import Path

from src.aas_od_config_converter import generate_config
from src.train_eval import trainer_aas as trainer
from src.train_eval import evaluator
from src.common import distributed_utils

# Anchor all I/O to this file's location, not the cwd, so the SAME command works
# from anywhere — the dashboard (cwd=OD_training) AND a cluster job launched from
# the repo root (e.g. `python ./OD_training/main.py ...` under srun).
OD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = OD_ROOT.parent

_WIN_ABS = re.compile(r'^[A-Za-z]:[/\\]')


def _resolve_aas(path):
    """Accept an AAS path that's absolute, repo-relative (AAS/...), OD-relative
    (../AAS/...), or cwd-relative — return the first that exists."""
    for cand in (Path(path), REPO_ROOT / path, OD_ROOT / path, Path.cwd() / path):
        if cand.exists():
            return str(cand)
    return path  # unchanged -> downstream raises a clear FileNotFoundError


def _translate_win_paths(config: dict) -> None:
    """On Linux (Docker), rewrite Windows host paths in config['paths'] to container mounts.
    Relies on SMARTSDG_WIN_ROOT / SMARTSDG_WIN_VAL_REAL env vars injected by server.py."""
    if os.name == 'nt':
        return
    win_root = os.environ.get('SMARTSDG_WIN_ROOT', '').replace('\\', '/')
    win_val = os.environ.get('SMARTSDG_WIN_VAL_REAL', '').replace('\\', '/')

    def translate(s: str) -> str:
        if not isinstance(s, str) or not _WIN_ABS.match(s):
            return s
        p = s.replace('\\', '/')
        if win_root and p.startswith(win_root):
            return '/workspace' + p[len(win_root):]
        if win_val and p.startswith(win_val):
            return '/valdata' + p[len(win_val):]
        return s

    if 'paths' in config:
        config['paths'] = {k: translate(v) for k, v in config['paths'].items()}


def _localize_paths(config: dict) -> None:
    """Make AAS paths work on THIS machine, wherever the passport was recorded.

    A passport recorded on Windows stores absolute `C:\\...` paths; on a Linux
    cluster those don't exist. Repo-internal data (`SmartSDG/Datasets/...`) is
    re-rooted under THIS repo (mounted at the run cwd on the cluster), and the
    external real-image val/test sets can be pointed at cluster locations via env
    vars. Lets the SAME passport + command train headlessly on the cluster with
    no edits."""
    paths = config.get('paths')
    if not isinstance(paths, dict):
        return

    def reroot(p):
        if not p:
            return p
        s = str(p).replace('\\', '/')
        if Path(s).exists():
            return s
        for anchor in ('SmartSDG/Datasets/', 'Datasets/yolo_experiments/', 'Datasets/'):
            i = s.find(anchor)
            if i >= 0 and (REPO_ROOT / s[i:]).exists():
                return str(REPO_ROOT / s[i:])
        return p

    for k in ('base_data_dir', 'val_data_path', 'test_data_path'):
        if k in paths:
            paths[k] = reroot(paths[k])
    # external real val/test on the cluster: env override > AAS value
    if os.environ.get('SMARTSDG_VAL_REAL'):
        paths['val_data_path'] = os.environ['SMARTSDG_VAL_REAL']
    if os.environ.get('SMARTSDG_TEST_REAL'):
        paths['test_data_path'] = os.environ['SMARTSDG_TEST_REAL']


def parse_sdg_metadata_from_path(data_path_string: str) -> dict:
    """
    Parses the final directory of a path to extract SDG metadata.

    Dataset names follow 'MODE_CLASS_LIGHT_ENV[_SUFFIX...]', where CLASS may
    itself contain underscores/hyphens and optional trailing tokens (e.g.
    'camera_noised') are skipped when scanning for LIGHT and ENV.
    """
    ENV_SUFFIXES = {'plane', 'withenv', 'withoutenv', 'noenv'}
    LIGHT_MODES = {'Low', 'Medium', 'High', 'S', 'M', 'L'}
    try:
        dir_name = Path(data_path_string.replace('\\', '/')).name.rstrip('/')
        parts = dir_name.split('_')
        metadata = {'raw_name': dir_name}
        metadata['mode'] = parts[0]
        if metadata['mode'] not in ('SINGLE', 'MULTI'):
            raise ValueError(f"Unknown mode '{metadata['mode']}' found in path.")

        # Scan tail from the end, skipping unknown tokens (e.g. 'camera_noised').
        tail = list(parts[1:])
        while tail and tail[-1] not in ENV_SUFFIXES and tail[-1] not in LIGHT_MODES:
            tail.pop()
        if tail and tail[-1] in ENV_SUFFIXES:
            metadata['env'] = tail.pop()
        else:
            metadata['env'] = 'noenv'
        if not tail or tail[-1] not in LIGHT_MODES:
            raise ValueError(f"Could not find a light mode in '{dir_name}'.")
        metadata['light'] = tail.pop()
        metadata['class'] = '_'.join(tail) if metadata['mode'] == 'SINGLE' else 'multi'
        return metadata
    except (IndexError, ValueError) as e:
        print(f"❌ FATAL ERROR: Could not parse SDG metadata from path: '{data_path_string}'")
        print(
            f"   Please ensure the directory name follows 'MODE_CLASS_LIGHT_ENV[_suffix]' or 'MODE_LIGHT_ENV[_suffix]'.")
        print(f"   Error details: {e}")
        exit(1)


def main(args):
    # --- 1. Setup Seeds ---
    distributed_utils.set_random_seeds(42)

    # --- 2. Load Base Configuration ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    _translate_win_paths(config)
    _localize_paths(config)   # re-root repo-internal data to THIS machine (cluster/headless)
    # Explicit CLI overrides win over the AAS value and SMARTSDG_VAL_REAL / _TEST_REAL.
    if getattr(args, 'data_dir', None):
        config.setdefault('paths', {})['base_data_dir'] = args.data_dir
    if getattr(args, 'val_dir', None):
        config.setdefault('paths', {})['val_data_path'] = args.val_dir
    if getattr(args, 'test_dir', None):
        config.setdefault('paths', {})['test_data_path'] = args.test_dir

    # --- 3. Dynamically Generate Paths (CRUCIAL FOR BOTH MODES) ---
    # This block MUST run for both 'train' and 'evaluate' modes to ensure
    # the config dictionary is fully populated with the correct paths.
    base_data_path = config['paths']['base_data_dir']
    sdg_meta = parse_sdg_metadata_from_path(base_data_path)

    # exp0 = void background, exp1 = flat plane, exp2 = full 3D environment
    env_to_id_map = {'noenv': 0, 'withoutenv': 0, 'plane': 1, 'withenv': 2}
    exp_id = env_to_id_map.get(sdg_meta['env'], 0)

    root_output_dir = OD_ROOT / "results"
    descriptive_run_name = sdg_meta['raw_name']

    # Generate the paths to the output directories
    new_custom_log_dir = root_output_dir / "experiment_outputs" / f"exp{exp_id}" / descriptive_run_name
    new_yolo_project_dir = root_output_dir / "yolo_run" / f"exp{exp_id}" / descriptive_run_name

    # Add the generated paths to the config dictionary
    config['paths']['custom_log_dir'] = str(new_custom_log_dir)
    config['paths']['yolo_project_dir'] = str(new_yolo_project_dir)

    # **THIS IS THE KEY THAT WAS MISSING**: Create the source directory key for the evaluator
    if 'evaluation' not in config: config['evaluation'] = {}  # Ensure 'evaluation' section exists
    config['evaluation']['source_yolo_runs_dir'] = str(new_yolo_project_dir)

    # --- 4. Save the final, fully populated config (optional but good practice) ---
    if distributed_utils.is_master_process():
        final_log_dir = Path(config['paths']['custom_log_dir'])
        final_log_dir.mkdir(parents=True, exist_ok=True)
        final_config_path = final_log_dir / 'final_run_config.yaml'
        with open(final_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"INFO: Final configuration paths generated and saved for this run.")

    # --- 5. Run the Requested Mode ---
    config['training']['cos_lr'] = args.cos_lr
    if args.patience is not None:
        config['training']['patience'] = args.patience
    mode = args.mode
    if mode == 'train':
        if distributed_utils.is_master_process():
            print("🚀 Starting Training Pipeline...")
        trainer.run_training_pipeline(config)
        # PASSPORT: self-document this CLI training run into the dataset AAS.
        if distributed_utils.is_master_process():
            try:
                repo_root = Path(__file__).resolve().parent.parent
                if str(repo_root) not in sys.path:
                    sys.path.insert(0, str(repo_root))
                from aas_passport import record_run
                ds_name = Path(base_data_path.replace('\\', '/').rstrip('/')).name
                train_cfg = {k: config.get(k) for k in
                             ("epochs", "batch_size", "img_size", "weights", "cameras_to_train")
                             if isinstance(config, dict) and config.get(k) is not None}
                meas = {}
                cands = list(Path(config['paths']['yolo_project_dir']).rglob('results.csv'))
                if cands:
                    import csv
                    rows = list(csv.DictReader(open(max(cands, key=lambda f: f.stat().st_mtime))))
                    if rows:
                        d = {k.strip(): v for k, v in rows[-1].items()}
                        for src, dst in (("metrics/mAP50(B)", "map50"),
                                         ("metrics/mAP50-95(B)", "map50_95")):
                            if d.get(src) not in (None, ""):
                                meas[dst] = round(float(d[src]), 4)
                p = record_run(ds_name, "Training", config=train_cfg, measurements=meas,
                               run_label="training", base_data_dir=base_data_path)
                print(f"--- AAS passport updated: {p} ---" if p else "--- AAS passport update skipped ---")
            except Exception as _e:
                print(f"--- AAS passport update skipped: {_e} ---")
    elif mode == 'evaluate':
        if distributed_utils.is_master_process():
            print("📊 Starting Evaluation Pipeline...")
        # Now, the 'config' dictionary is guaranteed to have the required key
        evaluator.run_evaluation_pipeline(config)
    else:
        if distributed_utils.is_master_process():
            print(f"❌ Error: Invalid mode '{mode}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YOLO Training and Evaluation with AAS Configuration.")

    parser.add_argument('--aas_file', type=str, required=False,
                        default=str(REPO_ROOT / "AAS" / "CA_Modul-OD_pipeline_v3.0.json"),
                        help='Path to the source AAS JSON file (absolute, or repo-relative like '
                             'AAS/datasets/<date>/CA_Modul-OD_dataset_<name>.json).')
    parser.add_argument('--config', type=str,
                        default=str(OD_ROOT / "config" / "generated_training_config.yaml"),
                        help='Path to save/read the generated YAML configuration file.')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'],
                        help='Run mode: train or evaluate.')
    parser.add_argument('--patience', type=int, default=None,
                        help='Early-stopping patience in epochs (default: Ultralytics default 100). Set 0 to disable.')
    parser.add_argument('--cos_lr', action='store_true', default=False,
                        help='Use cosine learning-rate schedule instead of linear.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training dataset folder (with camera subfolders). Overrides the AAS '
                             'base_data_dir — train any dataset (e.g. a camera-noised copy) with an '
                             'existing passport, no new AAS needed.')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Validation images folder (YOLO format). Overrides the AAS val_data_path '
                             'and SMARTSDG_VAL_REAL — handy on a cluster, e.g. '
                             '--val_dir /netscratch/.../data/val_new_2')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Test images folder for periodic evaluation. Overrides the AAS test_data_path.')

    cli_args = parser.parse_args()
    cli_args.aas_file = _resolve_aas(cli_args.aas_file)   # accept repo-/OD-/cwd-relative paths

    if distributed_utils.is_master_process():
        print("--- Step 1: Generating configuration from AAS... ---")
        print(f"    AAS: {cli_args.aas_file}")
        generate_config(aas_file_path=cli_args.aas_file, output_yaml_path=cli_args.config)
        print("-" * 50)

    print(f"--- Step 2: Running pipeline in '{cli_args.mode}' mode... ---")
    main(cli_args)