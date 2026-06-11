# run_from_aas.py (NO CHANGES NEEDED)
import sys
import subprocess
from pathlib import Path
import argparse

from cl_src.aas_cl_config_converter import AASConfigConverter

def main():
    parser = argparse.ArgumentParser(
        description="Run the Continual Learning pipeline using parameters from an AAS JSON file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config_file",
        type=Path,
        help="Path to the AAS JSON configuration file (e.g., 'CA_Modul-OD_pipeline_v0.7.json')."
    )
    args = parser.parse_args()

    try:
        print("--- Initializing AAS Converter ---")
        converter = AASConfigConverter(str(args.config_file))
        cl_params = converter.get_continual_learning_params(
            aas_id_short="CA_Modul",
            sm_id_short="OD_Module"
        )

        command = [
            sys.executable, 'main_cl.py',
            '--current_classes', cl_params['current_classes'],
            '--new_classes', cl_params['new_classes'],
            '--model_weights', cl_params['model_weights'],
            '--replay_data_dir', cl_params['replay_data_dir'],
            '--new_data_dir', cl_params['new_data_dir'],
            '--exp_name', cl_params['exp_name'],
            '--replay_ratio', str(cl_params['replay_ratio']),
            '--epochs', str(cl_params['epochs']),
            '--batch_size', str(cl_params['batch_size']),
            '--img_size', str(cl_params['img_size']),
        ]

        if 'output_dir' in cl_params:
            command.extend(['--output_dir', cl_params['output_dir']])
        if 'val_replay_data_dir' in cl_params:
            command.extend(['--val_replay_data_dir', cl_params['val_replay_data_dir']])
        if 'val_new_data_dir' in cl_params:
            command.extend(['--val_new_data_dir', cl_params['val_new_data_dir']])

        print("\n" + "=" * 27 + " Executing Command " + "=" * 27)
        print(" ".join(f'"{arg}"' if " " in arg else arg for arg in command))
        print("=" * 75 + "\n")

        subprocess.run(command, check=True, text=True, stderr=sys.stderr, stdout=sys.stdout)
        print("\n✅ Incremental learning step completed successfully!")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"❌ FATAL ERROR: {e}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ The training process failed with exit code {e.returncode}.")
        sys.exit(1)

if __name__ == '__main__':
    main()