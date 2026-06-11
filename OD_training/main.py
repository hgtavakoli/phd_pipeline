# cl_src/main.py (Corrected Version)

import argparse
import yaml
import os
from pathlib import Path

from src.aas_od_config_converter import generate_config
from src.train_eval import trainer_aas as trainer
from src.train_eval import evaluator
from src.common import distributed_utils


def parse_sdg_metadata_from_path(data_path_string: str) -> dict:
    """
    Parses the final directory of a path to extract SDG metadata.

    Dataset names follow 'MODE_CLASS_LIGHT_ENV', where CLASS may itself contain
    underscores/hyphens (e.g. component subsets like
    'MULTI_GlueGun-BLid-BTrailer_Medium_withenv'), so LIGHT and ENV are parsed
    from the END of the name, not by fixed position.
    """
    ENV_SUFFIXES = {'plane', 'withenv', 'withoutenv', 'noenv'}
    LIGHT_MODES = {'Low', 'Medium', 'High', 'S', 'M', 'L'}
    try:
        dir_name = Path(data_path_string).name
        parts = dir_name.split('_')
        metadata = {'raw_name': dir_name}
        metadata['mode'] = parts[0]
        if metadata['mode'] not in ('SINGLE', 'MULTI'):
            raise ValueError(f"Unknown mode '{metadata['mode']}' found in path.")

        metadata['env'] = parts[-1] if parts[-1] in ENV_SUFFIXES else 'noenv'
        light_part = parts[-2] if parts[-1] in ENV_SUFFIXES else parts[-1]
        if light_part not in LIGHT_MODES:
            raise ValueError(f"Could not find a light mode in '{dir_name}'.")
        metadata['light'] = light_part

        class_end = len(parts) - (2 if parts[-1] in ENV_SUFFIXES else 1)
        class_parts = parts[1:class_end]
        metadata['class'] = '_'.join(class_parts) if metadata['mode'] == 'SINGLE' else 'multi'
        return metadata
    except (IndexError, ValueError) as e:
        print(f"❌ FATAL ERROR: Could not parse SDG metadata from path: '{data_path_string}'")
        print(
            f"   Please ensure the final directory name follows the format 'MODE_CLASS_LIGHT_ENV' or 'MODE_LIGHT_ENV'.")
        print(f"   Error details: {e}")
        exit(1)


def main(args):
    # --- 1. Setup Seeds ---
    distributed_utils.set_random_seeds(42)

    # --- 2. Load Base Configuration ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- 3. Dynamically Generate Paths (CRUCIAL FOR BOTH MODES) ---
    # This block MUST run for both 'train' and 'evaluate' modes to ensure
    # the config dictionary is fully populated with the correct paths.
    base_data_path = config['paths']['base_data_dir']
    sdg_meta = parse_sdg_metadata_from_path(base_data_path)

    # exp0 = void background, exp1 = flat plane, exp2 = full 3D environment
    env_to_id_map = {'noenv': 0, 'withoutenv': 0, 'plane': 1, 'withenv': 2}
    exp_id = env_to_id_map.get(sdg_meta['env'], 0)

    root_output_dir = Path("results")
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
    mode = args.mode
    if mode == 'train':
        if distributed_utils.is_master_process():
            print("🚀 Starting Training Pipeline...")
        trainer.run_training_pipeline(config)
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

    parser.add_argument('--aas_file', type=str, required=False, default="../AAS/CA_Modul-OD_pipeline_v0.5.json",
                        help='Path to the source AAS JSON file.')
    parser.add_argument('--config', type=str, default="config/generated_training_config.yaml",
                        help='Path to save/read the generated YAML configuration file.')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'],
                        help='Run mode: train or evaluate.')

    cli_args = parser.parse_args()

    if distributed_utils.is_master_process():
        print("--- Step 1: Generating configuration from AAS... ---")
        generate_config(aas_file_path=cli_args.aas_file, output_yaml_path=cli_args.config)
        print("-" * 50)

    print(f"--- Step 2: Running pipeline in '{cli_args.mode}' mode... ---")
    main(cli_args)