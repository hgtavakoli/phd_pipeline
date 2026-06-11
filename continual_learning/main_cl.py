# main_cl.py
import argparse
from pathlib import Path
from cl_src.cl_orchestrator import run_incremental_step


def main():
    parser = argparse.ArgumentParser(
        description="Run a single step of Class-Incremental Object Detection training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Core Incremental Learning Parameters ---
    parser.add_argument('--current_classes', required=True, type=str,
                        help="Comma-separated string of class names the previous model already knows.")
    parser.add_argument('--new_classes', required=True, type=str,
                        help="Comma-separated string of new class names to learn in this step.")
    parser.add_argument('--model_weights', required=True, type=str,
                        help="Path to the 'best.pt' model from the previous training task.")

    # --- Dataset Path Parameters ---
    parser.add_argument('--replay_data_dir', required=True, type=str,
                        help="Path to the replay buffer directory (training set for old classes).")
    parser.add_argument('--new_data_dir', required=True, type=str,
                        help="Path to the dataset directory (training set for new classes).")

    # ================================================================= #
    # == NEW: Add arguments for the validation datasets (optional)     == #
    # ================================================================= #
    parser.add_argument('--val_replay_data_dir', type=str, default=None,
                        help="Path to the validation set for the old (replay) classes.")
    parser.add_argument('--val_new_data_dir', type=str, default=None,
                        help="Path to the validation set for the new classes.")

    # --- Training Configuration Parameters ---
    parser.add_argument('--output_dir', type=str, default="results/cl_runs",
                        help="Root directory to save the training outputs for this step.")
    parser.add_argument('--exp_name', type=str, default="incremental_step",
                        help="A descriptive name for this specific training run.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs for this step.")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size.")
    parser.add_argument('--img_size', type=int, default=640, help="Image size for training.")
    parser.add_argument('--replay_ratio', type=float, default=0.1,
                        help="The proportion of each batch that should be replay data (e.g., 0.4 for 40%).")
    args = parser.parse_args()

    # Convert comma-separated strings to lists
    args.current_classes = [c.strip() for c in args.current_classes.split(',')]
    args.new_classes = [c.strip() for c in args.new_classes.split(',')]

    print("--- Starting Incremental Learning Step ---")
    run_incremental_step(args)
    print("\n--- Incremental Learning Step Finished ---")


if __name__ == '__main__':
    main()


###########
    # python
    # main_cl.py - -current_classes
    # "Ylid, YTrailer, Cabin, Ggun,BLid, BTrailer, FChassis, Rchassis" - -new_classes
    # "Ylid2, YTrailer2, Cabin2, Ggun,BLid2, BTrailer2, FChassis1, Rchassis2" - -model_weights
    # "C:/Users/hota02/Documents/SDG/OD_training/yolo_runs_multi_mix/yolo_experiments_camera1/weights/best.pt" - -replay_data_dir
    # "C:/Users/hota02/Documents/sd_light_od/cluster_experiments/data/real_data" - -new_data_dir
    # "C:/Users/hota02/Documents/sd_light_od/cluster_experiments/data/truck_new" - -exp_name
    # "step3_learn_classD_E" - -val_replay_data_dir
    # "C:/Users/hota02/Documents/sd_light_od/cluster_experiments/data/val_limited" - -val_new_data_dir
    # "C:/Users/hota02/Documents/sd_light_od/cluster_experiments/data/val_real_v1_augmented/images"