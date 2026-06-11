import os
import math
import random
import time
import cv2
import numpy as np
import torch
import argparse  # NEW: Import argparse
from ultralytics import YOLO

# Configuration Constants (only file extensions remain)
IMG_EXTENSION = ".jpg"  # Or .png, etc.


# +--------------------------------------------------------------------------+
# |   Phase 2: Diversity Sampling (CCMS) & Clustering Helper Functions       |
# +--------------------------------------------------------------------------+

def cosine_similarity(A, B):
    """Calculates the cosine similarity between two vectors."""
    normA = np.linalg.norm(A)
    normB = np.linalg.norm(B)
    if normA == 0 or normB == 0:
        return 0.0
    return np.dot(A, B) / (normA * normB)


def compute_ccms(outputsA, outputsB):
    """
    Computes the Category-Conditioned Matching Similarity (CCMS) between two sets of predictions.
    'outputsA' and 'outputsB' are dictionaries with 'boxes', 'scores', and 'classes'.
    """
    predictions_number_A = len(outputsA['boxes'])
    predictions_number_B = len(outputsB['boxes'])

    if predictions_number_A == 0 or predictions_number_B == 0:
        return 0.0

    similarity_matrix = np.zeros((predictions_number_A, predictions_number_B))

    for i in range(predictions_number_A):
        for j in range(predictions_number_B):
            if outputsA['classes'][i] == outputsB['classes'][j]:
                boxA_i = outputsA['boxes'][i]
                boxB_j = outputsB['boxes'][j]
                similarity = cosine_similarity(boxA_i, boxB_j) + 1
                similarity_matrix[i, j] = similarity * outputsA['scores'][i]

    if np.sum(similarity_matrix) == 0:
        return 0.0

    scores_sum_A = np.sum(outputsA['scores'])
    scores_sum_B = np.sum(outputsB['scores'])
    max_similarity_AB = np.max(similarity_matrix, axis=1).sum() / scores_sum_A if scores_sum_A > 0 else 0
    max_similarity_BA = np.max(similarity_matrix, axis=0).sum() / scores_sum_B if scores_sum_B > 0 else 0
    ccms = (max_similarity_AB + max_similarity_BA) / 2
    return ccms


def k_centroid_greedy(dis_matrix, K):
    """Greedy algorithm to find initial centroids for K-Means."""
    N = dis_matrix.shape[0]
    c = np.random.randint(0, N, (1,))[0]
    centroids = [c]
    i = 1
    while i < K:
        centroids_diss = dis_matrix[:, centroids].copy()
        centroids_diss = centroids_diss.min(axis=1)
        centroids_diss[centroids] = -1
        new_c = np.argmax(centroids_diss)
        centroids.append(new_c)
        i += 1
    return centroids


def kmeans(dis_matrix, K, n_iter=100):
    """
    K-Means clustering implementation that uses a pre-computed distance matrix.
    Returns the indices of the final centroids.
    """
    N = dis_matrix.shape[0]
    centroids = k_centroid_greedy(dis_matrix, K)
    data_indices = np.arange(N)

    for _ in range(n_iter):
        centroid_dis = dis_matrix[:, centroids]
        cluster_assign = np.argmin(centroid_dis, axis=1)
        new_centroids = []
        for i in range(K):
            cluster_i_indices = data_indices[cluster_assign == i]
            if len(cluster_i_indices) == 0:
                new_centroids.extend(k_centroid_greedy(dis_matrix, 1))
                continue
            dis_mat_i = dis_matrix[cluster_i_indices][:, cluster_i_indices]
            new_centroid_local_index = np.argmin(dis_mat_i.sum(axis=1))
            new_centroid_global_index = cluster_i_indices[new_centroid_local_index]
            new_centroids.append(new_centroid_global_index)

        if np.array_equal(sorted(centroids), sorted(new_centroids)):
            break
        centroids = np.array(new_centroids)

    # CORRECTED: Robustly convert to list
    return list(centroids)


# +--------------------------------------------------------------------------+
# |      Phase 1: Uncertainty Sampling & Probabilistic Selection             |
# +--------------------------------------------------------------------------+

def probabilistic_selection(filename_value_dict, N):
    """Selects N keys from a dictionary based on their values as probabilities."""
    selected_keys = set()
    if N >= len(filename_value_dict):
        return list(filename_value_dict.keys())
    bias = 0.0
    while len(selected_keys) < N:
        for key, value in filename_value_dict.items():
            if key in selected_keys:
                continue
            if random.random() < (value + bias):
                selected_keys.add(key)
                if len(selected_keys) == N:
                    break
        bias += 0.01
    return list(selected_keys)


# +--------------------------------------------------------------------------+
# |                      Main PPAL Orchestration Function                    |
# +--------------------------------------------------------------------------+

# MODIFIED: Function signature now accepts the AL score threshold
def get_images_for_next_round(
        unlabeled_image_dir: str,
        yolo_model_path: str,
        num_classes: int,
        target_number: int,
        uncertainty_diversity_split: float = 0.5,
        al_score_threshold: float = 0.10
):
    """
    Selects a subset of images from an unlabeled pool using the PPAL strategy with a YOLOv12 model.
    """
    print("\n--- Starting PPAL Image Selection for YOLOv12 ---")
    # ... print statements ...

    # --- 1. SETUP & INFERENCE ---
    print("\n[Step 1/4] Loading model and running inference on unlabeled images...")
    model = YOLO(yolo_model_path)
    image_files = [f for f in os.listdir(unlabeled_image_dir) if f.lower().endswith(IMG_EXTENSION)]
    total_images = len(image_files)

    if total_images <= target_number:
        print("Warning: Target number is greater than or equal to total images. Returning all images.")
        return [os.path.splitext(f)[0] for f in image_files]

    all_outputs = {}
    for i, filename in enumerate(image_files):
        print(f"  - Inferring on image {i + 1}/{total_images}: {filename}", end='\r')
        image_path = os.path.join(unlabeled_image_dir, filename)
        try:
            # MODIFIED: Use the passed-in threshold
            results = model(image_path, verbose=False, conf=al_score_threshold)[0]
            basename = os.path.splitext(filename)[0]
            boxes = results.boxes.xywh.cpu().numpy().reshape(-1, 4) if results.boxes.xywh is not None else np.array([])
            scores = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else np.array([])
            classes = results.boxes.cls.cpu().numpy().astype(int) if results.boxes.cls is not None else np.array([])
            all_outputs[basename] = {"boxes": boxes, "scores": scores, "classes": classes}
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
    print("\nInference complete.")

    # --- 2. PHASE 1: DIFFICULTY-CALIBRATED UNCERTAINTY SAMPLING ---
    print("\n[Step 2/4] Calculating difficulty-calibrated uncertainty for each image...")
    class_uncertainties = {i: 1.0 for i in range(num_classes)}
    image_uncertainties = {}
    for basename, output in all_outputs.items():
        entropy = 0.0
        for i in range(len(output['scores'])):
            score = output['scores'][i]
            class_id = output['classes'][i]
            current_uncertainty = -score * math.log2(score + 1e-9)
            class_uncertainties[class_id] += current_uncertainty * 0.1
            entropy += current_uncertainty * class_uncertainties[class_id]
        image_uncertainties[basename] = entropy

    max_uncertainty = max(image_uncertainties.values(), default=1)
    if max_uncertainty > 0:
        for key in image_uncertainties:
            image_uncertainties[key] /= max_uncertainty

    num_for_uncertainty_pass = int((total_images - target_number) * (1 - uncertainty_diversity_split) + target_number)
    num_for_uncertainty_pass = min(num_for_uncertainty_pass, total_images)
    print(f"Selecting {num_for_uncertainty_pass} candidates based on uncertainty...")
    uncertainty_chosen_images = probabilistic_selection(image_uncertainties, num_for_uncertainty_pass)

    if len(uncertainty_chosen_images) <= target_number:
        print("Uncertainty pass resulted in fewer images than target. Returning selection.")
        return uncertainty_chosen_images[:target_number]

    # --- 3. PHASE 2: CCMS (DIVERSITY) ---
    print("\n[Step 3/4] Calculating similarity matrix for diversity sampling...")
    diversity_candidates = {name: all_outputs[name] for name in uncertainty_chosen_images}
    candidate_names = list(diversity_candidates.keys())
    candidate_outputs = list(diversity_candidates.values())
    count_for_diversity = len(candidate_names)
    similarity_matrix = np.zeros((count_for_diversity, count_for_diversity))
    start_time = time.time()
    for i in range(count_for_diversity):
        print(f"  - Computing similarities for image {i + 1}/{count_for_diversity}...", end='\r')
        for j in range(i, count_for_diversity):
            if i == j:
                continue
            similarity = compute_ccms(candidate_outputs[i], candidate_outputs[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    end_time = time.time()
    print(f"\nSimilarity matrix calculation took {end_time - start_time:.2f} seconds.")

    distance_matrix = 1 / (similarity_matrix + 0.2)
    np.fill_diagonal(distance_matrix, 0)

    # --- 4. K-MEANS CLUSTERING ---
    print("\n[Step 4/4] Running K-Means clustering to select final diverse set...")
    final_indices = kmeans(distance_matrix, K=target_number)
    final_image_selection = [candidate_names[idx] for idx in final_indices]
    print(f"\n--- PPAL Selection Complete. Selected {len(final_image_selection)} images. ---")
    return final_image_selection


# +--------------------------------------------------------------------------+
# |                         NEW: Command-Line Interface                      |
# +--------------------------------------------------------------------------+

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Select images for active learning using the PPAL method with a YOLOv12 model."
    )

    # Required arguments
    parser.add_argument('-i', '--unlabeled-dir', type=str, required=False,
                        default="C:/Users/hota02/Documents/SDG/active_learning/data/unlabled", help="Path to the directory containing unlabeled images.")
    parser.add_argument('-m', '--model-path', type=str, required=False,
                        default="C:/Users/hota02/Documents/SDG/OD_training/yolo_runs_multi_mix/yolo_experiments_camera1/weights/best.pt",
                        help="Path to the trained YOLOv12 .pt model file.")
    parser.add_argument('-n', '--num-classes', type=int, required=True,
                        help="The total number of classes the model was trained on.")
    parser.add_argument('-t', '--target-number', type=int, required=True,
                        help="The final number of images to select for annotation.")

    # Optional hyperparameters
    parser.add_argument('-s', '--split-ratio', type=float, default=0.5,
                        help="Uncertainty/Diversity split ratio (0.0 to 1.0). Default: 0.5.")
    parser.add_argument('-c', '--conf-threshold', type=float, default=0.10,
                        help="Confidence threshold for YOLO model during AL inference. Default: 0.10.")

    args = parser.parse_args()

    # Validate the split ratio
    if not 0.0 <= args.split_ratio <= 1.0:
        raise ValueError("Split ratio must be between 0.0 and 1.0.")

    try:
        selected_files = get_images_for_next_round(
            unlabeled_image_dir=args.unlabeled_dir,
            yolo_model_path=args.model_path,
            num_classes=args.num_classes,
            target_number=args.target_number,
            uncertainty_diversity_split=args.split_ratio,
            al_score_threshold=args.conf_threshold
        )

        print("\nFinal selected image basenames:")
        print(selected_files)

        print("\nNext step: Move these files to a 'to_be_labeled' folder for annotation.")

    except FileNotFoundError:
        print(f"\nERROR: Model file not found at '{args.model_path}'. Please check the path.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
