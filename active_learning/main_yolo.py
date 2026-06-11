# main_yolo.py (Refactored Version)

import os
import random
import yaml
import numpy as np
import torch
import shutil
from ultralytics import YOLO

# The utils file will also need to be refactored to accept the config dict
import utils_yolo as utils


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dataset_yaml(config: dict, data_path: str, round_num: int):
    """Creates the dataset.yaml file needed for YOLO training each round."""
    yaml_path = os.path.join(data_path, f"round_{round_num}.yaml")

    # Use the passed-in config dictionary instead of the old config file
    train_img_path_relative = os.path.join(config['data_dir'], config['F_WORKDIR'], f"round_{round_num}", "images",
                                           "train")
    val_img_path_relative = os.path.join(config['data_dir'], config['F_WORKDIR'], f"round_{round_num}", "images", "val")

    train_img_path_absolute = os.path.abspath(train_img_path_relative)
    val_img_path_absolute = os.path.abspath(val_img_path_relative)

    with open(config['dataset_yaml'], 'r') as f:
        data_yaml = yaml.safe_load(f)

    data_yaml['train'] = train_img_path_absolute.replace("\\", "/")
    data_yaml['val'] = val_img_path_absolute.replace("\\", "/")

    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    return yaml_path


# --- THIS IS THE NEW FUNCTION THAT YOUR LAUNCHER SCRIPT CAN IMPORT ---
def run_al_pipeline(config: dict):
    """
    This function contains the entire active learning loop and is driven
    by a configuration dictionary passed from the AAS parser.
    """
    seed_everything(config['SEED'])

    # Pass the config dictionary to the utility functions
    workdir_path, unlabeled_pool_path = utils.setup_workspace(config)
    model_path_for_round = config['initial_model_path']
    print(f"Loaded initial model path from AAS config: {model_path_for_round}")
    # --- ROUND 0: Initial Training ---
    print("\n" + "=" * 50)
    print("### STARTING INITIAL TRAINING - ROUND 0 ###")
    print("=" * 50)


    round_0_path = utils.create_round_dataset(config, workdir_path, 0)
    all_unlabeled_files = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(unlabeled_pool_path, "images"))]
    initial_files = random.sample(all_unlabeled_files, config['initial_subset_size'])
    utils.move_files(config, initial_files, unlabeled_pool_path, round_0_path)

    round_0_yaml = create_dataset_yaml(config, workdir_path, 0)

    model = YOLO(model_path_for_round)
    model.train(
        data=round_0_yaml,
        epochs=config['epochs_per_round'],
        batch=config['batch_size'],
        imgsz=config['img_size'],
        project=config['output_dir'],
        name=f"{config['PROJECT_NAME']}_round_0",
        seed=config['SEED'],
        deterministic=True
        # You can add more training args from config here
        # lr0=config['learning_rate'],
        # weight_decay=config['weight_decay'],
    )
    last_model_path = os.path.join(config['output_dir'], f"{config['PROJECT_NAME']}_round_0", "weights", "last.pt")

    # --- AL ROUNDS 1 to N ---
    for i in range(1, config['al_rounds'] + 1):
        print("\n" + "=" * 50)
        print(f"### STARTING ACTIVE LEARNING - ROUND {i} ###")
        print("=" * 50)

        selected_files = utils.get_images_for_next_round(
            config,  # Pass the config dict
            unlabeled_image_dir=unlabeled_pool_path,
            yolo_model_path=last_model_path
        )

        round_i_path = utils.create_round_dataset(config, workdir_path, i)

        prev_train_images = os.path.join(workdir_path, f"round_{i - 1}", "images", "train")
        curr_train_images = os.path.join(round_i_path, "images", "train")
        shutil.copytree(prev_train_images, curr_train_images, dirs_exist_ok=True)

        prev_train_labels = os.path.join(workdir_path, f"round_{i - 1}", "labels", "train")
        curr_train_labels = os.path.join(round_i_path, "labels", "train")
        shutil.copytree(prev_train_labels, curr_train_labels, dirs_exist_ok=True)

        utils.move_files(config, selected_files, unlabeled_pool_path, round_i_path)

        round_i_yaml = create_dataset_yaml(config, workdir_path, i)

        model = YOLO(last_model_path)
        model.train(
            data=round_i_yaml,
            epochs=config['epochs_per_round'],
            batch=config['batch_size'],
            imgsz=config['img_size'],
            project=config['output_dir'],
            name=f"{config['PROJECT_NAME']}_round_{i}",
            seed=config['SEED'],
            deterministic=True
        )
        last_model_path = os.path.join(config['output_dir'], f"{config['PROJECT_NAME']}_round_{i}", "weights",
                                       "last.pt")

    print("\n--- Active Learning Experiment Complete ---")