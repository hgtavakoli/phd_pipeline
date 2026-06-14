# utils_yolo.py (Corrected and Final Version)

import os
import shutil
import random
import math
import time
import numpy as np
from ultralytics import YOLO


# NOTE: The 'import config_yolo as config' line has been removed.

# +--------------------------------------------------------------------------+
# |                       Data Management Functions                          |
# +--------------------------------------------------------------------------+

def setup_workspace(config: dict):
    """Prepares the workspace for a new active learning experiment."""
    print("--- Setting up workspace ---")
    workdir_path = os.path.join(config['data_dir'], config['F_WORKDIR'])
    if os.path.exists(workdir_path):
        print(f"Removing existing workspace: {workdir_path}")
        shutil.rmtree(workdir_path)

    # Define source paths using dictionary access
    source_images_path = os.path.join(config['data_dir'], config['F_FULL_DATA'], "images", config['F_TRAIN'])
    source_labels_path = os.path.join(config['data_dir'], config['F_FULL_DATA'], "labels", config['F_TRAIN'])

    # Define destination paths using dictionary access
    unlabeled_pool_path = os.path.join(workdir_path, config['F_UNLABELED'])
    dest_images_path = os.path.join(unlabeled_pool_path, "images")
    dest_labels_path = os.path.join(unlabeled_pool_path, "labels")

    print(f"Copying training images to unlabeled pool: {dest_images_path}")
    shutil.copytree(source_images_path, dest_images_path)

    print(f"Copying training labels to unlabeled pool: {dest_labels_path}")
    shutil.copytree(source_labels_path, dest_labels_path)

    print("Workspace setup complete.")
    return workdir_path, unlabeled_pool_path


def create_round_dataset(config: dict, workdir_path: str, round_num: int):
    """Creates the directory structure for a new training round."""
    round_path = os.path.join(workdir_path, f"round_{round_num}")
    os.makedirs(os.path.join(round_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(round_path, "labels", "train"), exist_ok=True)

    # Use shutil.copytree for better cross-platform compatibility (avoids Windows symlink issues)
    val_images_src = os.path.abspath(os.path.join(config['data_dir'], config['F_FULL_DATA'], "images", config['F_VAL']))
    val_labels_src = os.path.abspath(os.path.join(config['data_dir'], config['F_FULL_DATA'], "labels", config['F_VAL']))

    val_images_dest = os.path.join(round_path, "images", "val")
    val_labels_dest = os.path.join(round_path, "labels", "val")

    shutil.copytree(val_images_src, val_images_dest)
    shutil.copytree(val_labels_src, val_labels_dest)

    return round_path


def move_files(config: dict, filenames: list, source_dir: str, dest_dir: str):
    """Moves image and label files from a source to a destination."""
    for name in filenames:
        # Move Image using dictionary access for extensions
        shutil.move(
            os.path.join(source_dir, "images", f"{name}{config['IMG_EXTENSION']}"),
            os.path.join(dest_dir, "images", "train", f"{name}{config['IMG_EXTENSION']}")
        )
        # Move Label using dictionary access for extensions
        shutil.move(
            os.path.join(source_dir, "labels", f"{name}{config['ANN_EXTENSION']}"),
            os.path.join(dest_dir, "labels", "train", f"{name}{config['ANN_EXTENSION']}")
        )


# +--------------------------------------------------------------------------+
# |  Active Learning Selection Strategies                                    |
# |                                                                          |
# |  config['al_strategy'] chooses how the next round's images are picked:   |
# |    random    – uniform random (the baseline every AL method must beat)   |
# |    entropy   – uncertainty only (PPAL phase 1, top-K)                    |
# |    ppal_box  – PPAL: uncertainty + CCMS diversity on BOX GEOMETRY        |
# |                (original implementation of this repo)                    |
# |    ppal_feat – PPAL: uncertainty + diversity on APPEARANCE EMBEDDINGS    |
# |                from the detector backbone (faithful to the PPAL paper,   |
# |                recommended)                                              |
# |  config['al_class_balance'] (bool) – boost images containing classes     |
# |  that are still rare in the labeled set.                                 |
# +--------------------------------------------------------------------------+

def cosine_similarity(A, B):
    normA, normB = np.linalg.norm(A), np.linalg.norm(B)
    return np.dot(A, B) / (normA * normB) if normA > 0 and normB > 0 else 0.0


def compute_ccms(outputsA, outputsB):
    n_A, n_B = len(outputsA['boxes']), len(outputsB['boxes'])
    if n_A == 0 or n_B == 0: return 0.0
    sim_matrix = np.zeros((n_A, n_B))
    for i in range(n_A):
        for j in range(n_B):
            if outputsA['classes'][i] == outputsB['classes'][j]:
                sim = cosine_similarity(outputsA['boxes'][i], outputsB['boxes'][j]) + 1
                sim_matrix[i, j] = sim * outputsA['scores'][i]
    sum_A, sum_B = np.sum(outputsA['scores']), np.sum(outputsB['scores'])
    max_sim_AB = np.max(sim_matrix, axis=1).sum() / sum_A if sum_A > 0 else 0
    max_sim_BA = np.max(sim_matrix, axis=0).sum() / sum_B if sum_B > 0 else 0
    return (max_sim_AB + max_sim_BA) / 2


def k_centroid_greedy(dis_matrix, K):
    N = dis_matrix.shape[0]
    centroids = [np.random.randint(0, N)]
    for _ in range(1, K):
        min_dists = np.min(dis_matrix[:, centroids], axis=1)
        min_dists[centroids] = -1
        centroids.append(np.argmax(min_dists))
    return centroids


def kmeans(dis_matrix, K, n_iter=100):
    N = dis_matrix.shape[0]
    centroids = k_centroid_greedy(dis_matrix, K)
    for _ in range(n_iter):
        cluster_assign = np.argmin(dis_matrix[:, centroids], axis=1)
        new_centroids = [
            i[np.argmin(dis_matrix[i][:, i].sum(axis=1))]
            for i in [np.where(cluster_assign == c)[0] for c in range(K)]
            if len(i) > 0
        ]
        if sorted(centroids) == sorted(new_centroids): break
        centroids = new_centroids
    return centroids


def probabilistic_selection(scores, N):
    if N >= len(scores): return list(scores.keys())
    selected, bias = set(), 0.0
    while len(selected) < N:
        for key, value in scores.items():
            if key not in selected and random.random() < (value + bias):
                selected.add(key)
                if len(selected) == N: break
        bias += 0.01
    return list(selected)


def _run_inference(config, model, unlabeled_image_dir, image_files):
    """Detector predictions for every pool image (boxes, scores, classes)."""
    all_outputs = {}
    for filename in image_files:
        results = model(os.path.join(unlabeled_image_dir, "images", filename),
                        verbose=False, conf=config['al_score_threshold'])[0]
        basename = os.path.splitext(filename)[0]
        all_outputs[basename] = {
            "boxes": results.boxes.xywh.cpu().numpy().reshape(-1, 4) if results.boxes.xywh is not None else np.array([]),
            "scores": results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else np.array([]),
            "classes": results.boxes.cls.cpu().numpy().astype(int) if results.boxes.cls is not None else np.array([]),
        }
    return all_outputs


def _uncertainty_scores(config, all_outputs, labeled_class_counts=None):
    """PPAL phase 1: difficulty-calibrated entropy per image, normalized to
    [0, 1]. Optionally boosts images containing classes that are rare in the
    labeled set so far (class-balance term for imbalanced real-world pools)."""
    class_uncertainties = {i: 1.0 for i in range(config['num_classes'])}
    image_uncertainties = {}
    for basename, output in all_outputs.items():
        entropy = 0.0
        for i in range(len(output['scores'])):
            score = output['scores'][i]
            if 0 <= output['classes'][i] < config['num_classes']:
                class_id = output['classes'][i]
                current_uncertainty = -score * math.log2(score + 1e-9)
                class_uncertainties[class_id] += current_uncertainty * 0.1
                entropy += current_uncertainty * class_uncertainties[class_id]
        image_uncertainties[basename] = entropy

    if config.get('al_class_balance') and labeled_class_counts:
        # rarity weight: classes barely present in the labeled set get up to a
        # 50% bonus for images that contain them
        total = sum(labeled_class_counts.values()) or 1
        rarity = {c: 1.0 - (n / total) for c, n in labeled_class_counts.items()}
        for basename, output in all_outputs.items():
            present = set(int(c) for c in output['classes'])
            if present:
                bonus = max(rarity.get(c, 1.0) for c in present)
                image_uncertainties[basename] *= (1.0 + 0.5 * bonus)

    max_unc = max(image_uncertainties.values() or [1])
    if max_unc <= 0:
        # model produced no confident detections yet (common very early) —
        # fall back to uniform scores so selection degrades to random, not crash
        print("  (no confident detections; uncertainty scores uniform this round)")
        return {k: 1.0 for k in image_uncertainties}
    return {k: v / max_unc for k, v in image_uncertainties.items()}


def _phase1_candidates(config, scores, total_images, target_number):
    """Probabilistic uncertainty pre-selection (PPAL phase 1 output size)."""
    split = config['ud_split']
    num_for_unc_pass = int((total_images - target_number) * (1 - split) + target_number)
    return probabilistic_selection(scores, num_for_unc_pass)


def _diversity_box(all_outputs, cand_names, target_number):
    """Original repo behavior: CCMS on bounding-box geometry vectors."""
    cand_outputs = [all_outputs[n] for n in cand_names]
    n = len(cand_names)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_ccms(cand_outputs[i], cand_outputs[j])
            sim_matrix[i, j] = sim_matrix[j, i] = sim
    dist_matrix = 1 / (sim_matrix + 0.2)
    np.fill_diagonal(dist_matrix, 0)
    final_indices = kmeans(dist_matrix, K=target_number)
    return [cand_names[i] for i in final_indices]


def _diversity_features(config, model, unlabeled_image_dir, cand_names, target_number):
    """Faithful PPAL phase 2: diversity on APPEARANCE EMBEDDINGS from the
    detector backbone (ultralytics model.embed), cosine distance + k-means.
    Visually similar images cluster together; one representative per cluster
    is selected."""
    paths = [os.path.join(unlabeled_image_dir, "images", f"{n}{config['IMG_EXTENSION']}")
             for n in cand_names]
    embeddings = []
    batch = 16
    for i in range(0, len(paths), batch):
        embs = model.embed(paths[i:i + batch], verbose=False)
        embeddings.extend(e.cpu().numpy().flatten() for e in embs)
    X = np.stack(embeddings)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sim = X @ X.T                       # cosine similarity in [-1, 1]
    dist_matrix = 1.0 - sim             # cosine distance
    np.fill_diagonal(dist_matrix, 0)
    final_indices = kmeans(dist_matrix, K=target_number)
    return [cand_names[i] for i in final_indices]


def get_images_for_next_round(config: dict, unlabeled_image_dir: str, yolo_model_path: str,
                              labeled_class_counts: dict = None):
    """Select the next batch of images from the unlabeled pool, using the
    strategy in config['al_strategy'] (default: ppal_feat)."""
    strategy = config.get('al_strategy', 'ppal_feat')
    target_number = config['add_per_round']
    image_files = [f for f in os.listdir(os.path.join(unlabeled_image_dir, "images"))
                   if f.lower().endswith(config['IMG_EXTENSION'])]
    total_images = len(image_files)
    print(f"\n--- AL selection: strategy='{strategy}', pool={total_images}, budget={target_number} ---")

    if total_images <= target_number:
        print("Warning: Not enough images in pool. Returning all remaining images.")
        return [os.path.splitext(f)[0] for f in image_files]

    if strategy == "random":
        return random.sample([os.path.splitext(f)[0] for f in image_files], target_number)

    model = YOLO(yolo_model_path)
    print(f"[1/4] Running inference on {total_images} images...")
    all_outputs = _run_inference(config, model, unlabeled_image_dir, image_files)

    print("[2/4] Calculating uncertainty...")
    scores = _uncertainty_scores(config, all_outputs, labeled_class_counts)

    if strategy == "entropy":
        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[:target_number]

    cand_names = _phase1_candidates(config, scores, total_images, target_number)
    if len(cand_names) <= target_number:
        return cand_names

    print(f"[3/4] Diversity stage on {len(cand_names)} candidates...")
    if strategy == "ppal_feat":
        final_selection = _diversity_features(config, model, unlabeled_image_dir, cand_names, target_number)
    else:  # ppal_box
        final_selection = _diversity_box(all_outputs, cand_names, target_number)

    print(f"[4/4] --- Selection Complete. Selected {len(final_selection)} images. ---")
    return final_selection


def count_labeled_classes(labels_dir: str, num_classes: int) -> dict:
    """Class-instance counts across all YOLO label files in a directory —
    used by the class-balance term to know which classes are still rare."""
    counts = {i: 0 for i in range(num_classes)}
    if not os.path.isdir(labels_dir):
        return counts
    for f in os.listdir(labels_dir):
        if f.endswith(".txt"):
            with open(os.path.join(labels_dir, f)) as fh:
                for line in fh:
                    parts = line.split()
                    if parts:
                        cid = int(parts[0])
                        if 0 <= cid < num_classes:
                            counts[cid] += 1
    return counts