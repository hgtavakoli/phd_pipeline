# cl_src/replay_buffer.py
import random
from pathlib import Path
from collections import defaultdict


class ReplayBuffer:
    def __init__(self, config: dict):
        cl_config = config.get('ContinualLearning', {})
        self.buffer_size_per_class = cl_config.get('replay_buffer_size_per_class', 20)
        self.strategy = cl_config.get('exemplar_selection_strategy', 'random')
        self.exemplars = defaultdict(list)
        print(f"  - Replay Buffer initialized. Size per class: {self.buffer_size_per_class}, Strategy: {self.strategy}")

    def update(self, new_classes: list, new_task_base_path: Path, cameras: list):
        """
        Adds new exemplars by searching through all specified camera folders
        within a new task's base directory.
        """
        print(f"  - Updating replay buffer with exemplars for: {new_classes}")

        all_candidate_images = []
        for camera in cameras:
            camera_path = new_task_base_path / camera
            if camera_path.is_dir():
                all_candidate_images.extend(list(camera_path.glob('*.png')))
                all_candidate_images.extend(list(camera_path.glob('*.jpg')))

        if not all_candidate_images:
            print("  - WARNING: No images found in any camera folder to update the replay buffer.")
            return

        for class_name in new_classes:
            # For simplicity, we sample from the combined list of all images from all cameras.
            # A more advanced version could ensure balanced sampling per camera.
            num_to_sample = min(self.buffer_size_per_class, len(all_candidate_images))
            selected_exemplars = random.sample(all_candidate_images, num_to_sample)
            self.exemplars[class_name] = [str(p.resolve()) for p in selected_exemplars]

        total_exemplars = sum(len(v) for v in self.exemplars.values())
        print(f"  - ✅ Buffer update complete. Total exemplars stored: {total_exemplars}")

    def get_exemplar_paths(self) -> list:
        """Returns a flat list of all image paths currently stored in the buffer."""
        all_paths = []
        for class_name, paths in self.exemplars.items():
            all_paths.extend(paths)
        return all_paths