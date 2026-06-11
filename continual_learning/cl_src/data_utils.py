# cl_src/data_utils.py
import os
import shutil
import random
import torch
from pathlib import Path
from torch.utils.data.sampler import Sampler
from typing import Sized


def prepare_task_dataset(new_data_dir: Path, replay_data_dir: Path, output_dir: Path, class_offset: int):
    """
    Creates a temporary, merged dataset for a single training step.
    It copies old data, copies and re-labels new data, and returns lists of filenames.

    Returns:
        tuple: A tuple containing (
            Path to the merged dataset,
            list of new image filenames,
            list of replay image filenames
        )
    """
    if not new_data_dir.is_dir():
        raise FileNotFoundError(f"New data directory not found: {new_data_dir}")
    if not replay_data_dir.is_dir():
        raise FileNotFoundError(f"Replay data directory not found: {replay_data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  - Preparing merged dataset in: '{output_dir}'")

    # --- 1. Process and RE-LABEL the new task data ---
    new_task_images_paths = list(new_data_dir.glob('*.png')) + list(new_data_dir.glob('*.jpg'))
    new_task_image_names = [p.name for p in new_task_images_paths]
    print(f"  - Found {len(new_task_images_paths)} images in new dataset.")
    print(f"  - Applying class index offset of +{class_offset} to new data labels.")

    for img_path in new_task_images_paths:
        shutil.copy(img_path, output_dir)
        label_path = img_path.with_suffix('.txt')
        if not label_path.exists():
            continue

        new_label_lines = []
        with open(label_path, 'r') as f_in:
            for line in f_in:
                parts = line.strip().split()
                if not parts: continue
                original_idx = int(parts[0])
                new_idx = original_idx + class_offset
                new_label_lines.append(f"{new_idx} {' '.join(parts[1:])}\n")

        with open(output_dir / label_path.name, 'w') as f_out:
            f_out.writelines(new_label_lines)

    # --- 2. Process and COPY the replayed exemplar data ---
    replay_images_paths = list(replay_data_dir.glob('*.png')) + list(replay_data_dir.glob('*.jpg'))
    replay_image_names = [p.name for p in replay_images_paths]
    print(f"  - Found {len(replay_images_paths)} images in replay buffer.")
    for img_path in replay_images_paths:
        shutil.copy(img_path, output_dir)
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            shutil.copy(label_path, output_dir)

    print(f"  - ✅ Merged dataset created successfully.")
    return output_dir, new_task_image_names, replay_image_names

# ==================================================================== #
# == THIS IS THE FUNCTION THAT WAS MISSING FROM YOUR FILE             == #
# ==================================================================== #
def prepare_task_validation_set(new_val_dir: Path, replay_val_dir: Path, output_dir: Path, class_offset: int):
    """
    Creates a temporary, merged VALIDATION dataset. It copies old validation
    data and copies+re-labels new validation data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  - Preparing merged VALIDATION dataset in: '{output_dir}'")

    # Process and RE-LABEL the new validation data
    if new_val_dir and new_val_dir.is_dir():
        new_val_images = list(new_val_dir.glob('*.png')) + list(new_val_dir.glob('*.jpg'))
        for img_path in new_val_images:
            shutil.copy(img_path, output_dir)
            label_path = img_path.with_suffix('.txt')
            if not label_path.exists(): continue
            new_label_lines = []
            with open(label_path, 'r') as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    if not parts: continue
                    new_idx = int(parts[0]) + class_offset
                    new_label_lines.append(f"{new_idx} {' '.join(parts[1:])}\n")
            with open(output_dir / label_path.name, 'w') as f_out:
                f_out.writelines(new_label_lines)

    # COPY the old validation data
    if replay_val_dir and replay_val_dir.is_dir():
        replay_val_images = list(replay_val_dir.glob('*.png')) + list(replay_val_dir.glob('*.jpg'))
        for img_path in replay_val_images:
            shutil.copy(img_path, output_dir)
            label_path = img_path.with_suffix('.txt')
            if label_path.exists(): shutil.copy(label_path, output_dir)

    print(f"  - ✅ Merged validation set created.")
    return output_dir


class BalancedBatchSampler(Sampler):
    """
    Custom PyTorch sampler to create batches with a fixed ratio of new and replay data.
    This version is robust against edge cases from integer division.
    """

    def __init__(self, dataset: Sized, new_indices: list, replay_indices: list, batch_size: int, replay_ratio: float):
        super().__init__(dataset)

        if not 0.0 <= replay_ratio <= 1.0:
            raise ValueError(f"replay_ratio must be between 0.0 and 1.0, but got {replay_ratio}")

        self.new_indices = new_indices
        self.replay_indices = replay_indices
        self.batch_size = batch_size

        # --- FIX 1: Use rounding for more intuitive splits ---
        replay_samples = round(batch_size * replay_ratio)

        # --- FIX 2: Handle edge cases gracefully ---
        # If replay data exists, ensure we sample at least one replay image,
        # unless the user explicitly set the ratio to 0.
        if replay_indices and replay_samples == 0 and replay_ratio > 0:
            self.replay_samples_per_batch = 1
        else:
            self.replay_samples_per_batch = int(replay_samples)

        self.new_samples_per_batch = batch_size - self.replay_samples_per_batch

        # --- FIX 3: Add explicit validation to prevent silent failures or crashes ---
        if self.new_samples_per_batch <= 0:
            raise ValueError(
                f"Calculated new_samples_per_batch is {self.new_samples_per_batch}. "
                f"This is not allowed. Check your batch_size ({batch_size}) and replay_ratio ({replay_ratio}). "
                "The number of new samples per batch must be greater than 0."
            )

        if not self.new_indices:
            raise ValueError("Cannot create batches with an empty list of new_indices.")

        # Ensure we have enough replay samples, even if we need to repeat them
        if self.replay_indices and len(self.replay_indices) < self.replay_samples_per_batch:
            factor = math.ceil(self.replay_samples_per_batch / len(self.replay_indices))
            self.replay_indices = self.replay_indices * factor
            print(f"  - INFO: Replay buffer is smaller than required per-batch size. Replay indices will be repeated.")

    def __iter__(self):
        random.shuffle(self.new_indices)

        num_batches = len(self.new_indices) // self.new_samples_per_batch
        for i in range(num_batches):
            # Get a chunk of new data indices
            new_batch_part = self.new_indices[i * self.new_samples_per_batch: (i + 1) * self.new_samples_per_batch]

            # Get a random chunk of replay data indices
            if self.replay_indices and self.replay_samples_per_batch > 0:
                replay_batch_part = random.sample(self.replay_indices, self.replay_samples_per_batch)
            else:
                replay_batch_part = []

            # Combine and shuffle them to form the final batch
            batch = new_batch_part + replay_batch_part
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.new_indices) // self.new_samples_per_batch