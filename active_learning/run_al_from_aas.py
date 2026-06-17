"""
This script serves as the entry point for the Active Learning pipeline.
It reads the entire experiment configuration from a specified AAS file,
parses the specific 'ODOptimisation -> ActiveLearning' structure,
and then launches the AL process.
"""
import argparse
import json
import sys
from pathlib import Path

# Required for AAS parsing
from aas_core3 import jsonization
from aas_core3 import types as aas_types

# Import the main logic from your AL pipeline
# Assumes main_yolo.py contains a function `run_al_pipeline(config)`
from main_yolo import run_al_pipeline


class AASConfigConverter:
    """
    Parses a specific AAS structure to generate a flat configuration dictionary.
    Expected Structure:
    Submodel -> ODOptimisation -> ActiveLearning -> [Paths, TrainingParameters, ...]
    """
    def __init__(self, aas_file_path: str):
        self.aas_file_path = aas_file_path
        self.environment = self._load_environment()

    def _load_environment(self) -> aas_types.Environment:
        try:
            with open(self.aas_file_path, "r", encoding="utf-8") as f:
                jsonable = json.load(f)
            # This is the correct way to load from the provided JSON snippet
            # which might be a part of a larger AAS file.
            if "assetAdministrationShells" not in jsonable:
                 # Assuming the snippet is the 'value' of a Submodel
                 submodel_json = {"submodelElements": [jsonable]}
                 return jsonization.environment_from_jsonable({"submodels": [submodel_json]})
            return jsonization.environment_from_jsonable(jsonable)
        except FileNotFoundError:
            raise FileNotFoundError(f"AAS file not found at '{self.aas_file_path}'.")
        except Exception as e:
            raise RuntimeError(f"Error parsing AAS file: {e}")

    def _convert_value(self, value: str):
        """Converts string values to appropriate Python types."""
        # Clean quotes and whitespace from string values
        if isinstance(value, str):
            value = value.strip().strip('"')
            val_lower = value.lower()
            if val_lower == 'true': return True
            if val_lower == 'false': return False
            try: return int(value)
            except ValueError:
                try: return float(value)
                except ValueError: return value
        return value

    def get_config(self) -> dict:
        """
        Extracts the configuration by navigating the specific AAS structure.
        """
        # --- Step 1: Find the 'ODOptimisation' Collection ---
        # We search through all submodels to find the required element.
        od_optimisation_element = None
        for sm in self.environment.submodels:
            if not sm.submodel_elements: continue
            element = next((el for el in sm.submodel_elements if el.id_short == "ODOptimisation"), None)
            if element:
                od_optimisation_element = element
                break

        if not od_optimisation_element:
            raise ValueError("SubmodelElementCollection 'ODOptimisation' not found in any submodel.")

        # --- Step 2: Find the 'ActiveLearning' List within ODOptimisation ---
        if not od_optimisation_element.value:
            raise ValueError("'ODOptimisation' element is empty.")

        active_learning_element = next((el for el in od_optimisation_element.value if el.id_short == "ActiveLearning"), None)

        if not active_learning_element:
            raise ValueError("SubmodelElementList 'ActiveLearning' not found inside 'ODOptimisation'.")

        # --- Step 3: Parse all parameter groups within ActiveLearning ---
        final_config = {}
        if not active_learning_element.value:
            raise ValueError("'ActiveLearning' element is empty.")

        # The 'value' of ActiveLearning is a list of parameter groups (e.g., Paths, TrainingParameters)
        for param_group in active_learning_element.value:
            if not param_group.value: continue
            # The 'value' of each group is a list of the final properties
            for prop in param_group.value:
                if hasattr(prop, 'id_short') and hasattr(prop, 'value'):
                    key = prop.id_short
                    value = self._convert_value(prop.value)
                    final_config[key] = value

        return final_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Active Learning using an AAS configuration file.")
    parser.add_argument(
        '--aas_file',
        type=str,
        required=True,
        help='Path to the source AAS JSON file.'
    )
    args = parser.parse_args()

    try:
        # --- Step 1: Parse the AAS file to create a config dictionary ---
        print("--- Loading configuration from AAS ---")
        converter = AASConfigConverter(aas_file_path=args.aas_file)
        experiment_config = converter.get_config()
        print("✅ AAS configuration loaded successfully:")
        print(json.dumps(experiment_config, indent=2))

        # --- Step 2: Add static project constants to the config ---
        experiment_config['PROJECT_NAME'] = "yolo_al_experiment"
        experiment_config['F_FULL_DATA'] = "full-data"
        experiment_config['F_TRAIN'] = "train"
        experiment_config['F_VAL'] = "val"
        experiment_config['F_WORKDIR'] = "workspace"
        experiment_config['F_UNLABELED'] = "unlabeled"
        # SDG datasets are .png; allow override from the AAS (img_extension)
        experiment_config['IMG_EXTENSION'] = experiment_config.get('img_extension', '.png')
        experiment_config['ANN_EXTENSION'] = ".txt"
        experiment_config['SEED'] = int(experiment_config.get('seed', 42))
        # Strategy defaults (overridable from the AAS / dashboard)
        experiment_config.setdefault('al_strategy', 'ppal_feat')
        experiment_config.setdefault('al_class_balance', True)
        experiment_config.setdefault('al_score_threshold', 0.10)
        experiment_config.setdefault('ud_split', 0.5)

        # --- Step 3: Launch the Active Learning Pipeline ---
        print("\n--- Starting Active Learning Pipeline ---")
        run_al_pipeline(experiment_config)

        # --- Step 4: PASSPORT — self-document this CLI run into the dataset AAS ---
        try:
            from pathlib import Path as _P
            repo_root = _P(__file__).resolve().parent.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from aas_passport import record_run
            base = experiment_config.get('base_data_dir') or ''
            ds_name = (_P(base.replace('\\', '/').rstrip('/')).name if base
                       else _P(args.aas_file).stem.replace('CA_Modul-OD_dataset_', ''))
            al_cfg = {k: experiment_config.get(k) for k in
                      ('al_strategy', 'al_class_balance', 'al_rounds', 'initial_subset_size',
                       'add_per_round', 'epochs_per_round', 'al_score_threshold', 'ud_split')
                      if experiment_config.get(k) is not None}
            meas = {}
            try:
                import csv
                out = experiment_config.get('output_dir')
                cands = list(_P(out).rglob('al_summary.csv')) if out else []
                if cands:
                    rows = list(csv.reader(open(max(cands, key=lambda f: f.stat().st_mtime))))
                    if len(rows) >= 2:
                        meas['al_rounds_logged'] = len(rows) - 1
            except Exception:
                pass
            p = record_run(ds_name, "AL", config=al_cfg, measurements=meas,
                           run_label="AL", base_data_dir=(base or None))
            print(f"--- AAS passport updated: {p} ---" if p else "--- AAS passport update skipped ---")
        except Exception as _e:
            print(f"--- AAS passport update skipped: {_e} ---")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"❌ FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)