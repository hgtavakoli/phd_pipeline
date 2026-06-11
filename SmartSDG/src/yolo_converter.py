"""
YOLO conversion and dataset quality filtering.

Takes raw Isaac Sim Replicator output (RGB PNGs + bounding_box_2d_tight .npy/
.json annotations) and produces a flat YOLO-format dataset (image + .txt label
pairs), applying the quality filters described in docs/HYPERPARAMETERS.md.

PIPELINE INVARIANT — never keep an image containing an unlabeled object:
if any bounding box must be removed (occluded, border-cut, degenerate), the
default policies discard the ENTIRE frame. An image with a visible but
unlabeled object teaches the detector to ignore that object.
"""

import json
import os
import shutil
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from config_loader import AppConfig

# ==========================================================
# CONVERSION
# ==========================================================

def _convert_coords_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    """Convert pixel-space corner coordinates to normalized YOLO (cx, cy, w, h)."""
    box_width = float(x_max - x_min)
    box_height = float(y_max - y_min)
    x_center = float(x_min + (box_width / 2))
    y_center = float(y_min + (box_height / 2))
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = box_width / img_width
    height_norm = box_height / img_height
    return x_center_norm, y_center_norm, width_norm, height_norm


def run_yolo_conversion(source_dir: str, config: AppConfig, light_mode: str, min_visibility: float = 0.0,
                        drop_truncated: bool = False, truncation_policy: str = "box",
                        occlusion_policy: str = "box"):
    """
    Finds Isaac Sim output and converts it to YOLO format.
    The final output folder will be named after the source folder.

    min_visibility: keep a bounding box only if at least this fraction of the
    object is visible (1 - occlusionRatio from Replicator). NOTE: occlusionRatio
    only measures occlusion BY OTHER OBJECTS — it does not detect objects that
    are partially outside the camera frame. Use drop_truncated for that.

    drop_truncated: drop objects that touch the image border (i.e. the object is
    partially outside the camera's field of view).

    truncation_policy: what to do when a truncated object is found --
      "box"   : remove only that object's label (the object stays visible in
                the image but unlabeled!)
      "frame" : discard the whole frame (image + all labels). Use this when the
                dataset must contain no border-cut objects at all.

    Frames where no box survives the filters are skipped entirely.
    """
    print("\n" + "=" * 60)
    print("?? STARTING YOLO CONVERSION AND RESTRUCTURING")
    if min_visibility > 0.0:
        if occlusion_policy == "frame":
            print(f"  > Visibility filter: discarding WHOLE FRAMES containing any object < {min_visibility*100:.0f}% visible (occluded by other objects)")
        else:
            print(f"  > Visibility filter: keeping objects with >= {min_visibility*100:.0f}% visible (occlusion by other objects)"
                  " (NOTE: occluded object pixels stay in the image — use occlusion_policy 'frame' to discard such images entirely)")
    if drop_truncated:
        if truncation_policy == "frame":
            print("  > Truncation filter: discarding WHOLE FRAMES that contain any object touching the image border")
        else:
            print("  > Truncation filter: dropping labels of objects that touch the image border"
                  " (NOTE: the object pixels stay in the image — use truncation_policy 'frame' to discard such images entirely)")

    # UPDATED: Access the YOLO_CONVERSION object directly using attributes
    yolo_config = config.YOLO_CONVERSION

    # UPDATED: Use the pre-processed absolute path and direct attribute access
    dest_dir_base = yolo_config.output_dir_abs
    class_names = yolo_config.CLASS_NAMES

    # This parameter is not in your current config, so we provide a default.
    # You could add 'exp_number' to your YOLOConversionConfig dataclass to make it configurable.
    exp_number = "Exp0"

    if not all([dest_dir_base, class_names]):
        print("? ERROR: YOLO_CONVERSION settings missing in config.yaml.")
        return None

    root_dir = Path(source_dir)
    source_folder_name = root_dir.name
    dest_dir_final = Path(dest_dir_base) / source_folder_name

    print(f"  > Source: {source_dir}")
    print(f"  > Destination: {dest_dir_final}")
    print("=" * 60)

    # ... The rest of the function requires no changes ...

    data_folders_to_process = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "rgb" in dirnames and "bounding_box_2d_tight" in dirnames:
            data_folders_to_process.append(Path(dirpath))
            dirnames.clear()

    if not data_folders_to_process:
        print(f"? WARNING: No valid data folders found in '{root_dir}'.")
        return dest_dir_final

    print(f"? Found {len(data_folders_to_process)} data folder(s) to process.")
    isaac_to_yolo_map = {name.lower(): i for i, name in enumerate(class_names)}
    created_dirs = set()
    processed_count = 0
    stats = {"boxes_kept": 0, "boxes_dropped_occluded": 0, "boxes_dropped_truncated": 0,
             "boxes_dropped_degenerate": 0, "frames_skipped_empty": 0, "frames_dropped_truncated": 0,
             "frames_dropped_occluded": 0, "frames_dropped_degenerate": 0, "frames_total": 0, "frames_kept": 0}

    for rp_folder in data_folders_to_process:
        bbox_dir = rp_folder / "bounding_box_2d_tight"
        rgb_dir = rp_folder / "rgb"

        camera_id = "camera2" if "RenderProduct" in rp_folder.name and rp_folder.name.endswith(
            '_01') else rp_folder.name

        final_output_dir = dest_dir_final / camera_id

        if final_output_dir not in created_dirs:
            final_output_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.add(final_output_dir)
            print(f"\n?? Created/Targeting output directory: {final_output_dir}")

        print(f"  -> Processing source: {rp_folder.relative_to(root_dir)}")

        for npy_file in sorted(bbox_dir.glob("bounding_box_2d_tight_*.npy")):
            frame_number_str = npy_file.stem.split('_')[-1]
            image_path = next(rgb_dir.glob(f"rgb_{frame_number_str}.*"), None)
            json_path = npy_file.with_name(f"bounding_box_2d_tight_labels_{frame_number_str}.json")

            if not image_path or not json_path.exists():
                continue
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
                    img_ext = image_path.suffix

                with open(json_path, 'r') as f:
                    json_labels = json.load(f)

                bboxes_data = np.load(npy_file, allow_pickle=True)
                yolo_lines = []
                frame_has_truncated = False
                frame_has_occluded = False
                frame_has_degenerate = False
                stats["frames_total"] += 1

                if bboxes_data.size > 0:
                    for bbox_record in bboxes_data:
                        semantic_id_str = str(bbox_record['semanticId'])
                        if semantic_id_str in json_labels:
                            isaac_class_name = json_labels[semantic_id_str].get('class', '').lower()
                            if isaac_class_name in isaac_to_yolo_map:
                                if min_visibility > 0.0 and 'occlusionRatio' in (bboxes_data.dtype.names or ()):
                                    occ = float(bbox_record['occlusionRatio'])
                                    if np.isnan(occ):
                                        occ = 1.0  # invalid ratio -> treat as fully occluded
                                    if (1.0 - occ) < min_visibility:
                                        stats["boxes_dropped_occluded"] += 1
                                        frame_has_occluded = True
                                        continue
                                if bbox_record['x_max'] <= bbox_record['x_min'] or bbox_record['y_max'] <= bbox_record['y_min']:
                                    stats["boxes_dropped_degenerate"] += 1
                                    frame_has_degenerate = True
                                    continue  # degenerate (zero-area) box
                                if drop_truncated and (bbox_record['x_min'] <= 0 or bbox_record['y_min'] <= 0
                                                       or bbox_record['x_max'] >= img_width - 1
                                                       or bbox_record['y_max'] >= img_height - 1):
                                    stats["boxes_dropped_truncated"] += 1
                                    frame_has_truncated = True
                                    continue  # object partially outside the camera frame
                                yolo_class_id = isaac_to_yolo_map[isaac_class_name]
                                yolo_coords = _convert_coords_to_yolo(bbox_record['x_min'], bbox_record['y_min'],
                                                                      bbox_record['x_max'], bbox_record['y_max'],
                                                                      img_width, img_height)
                                yolo_line = f"{yolo_class_id} " + " ".join(f"{c:.6f}" for c in yolo_coords)
                                yolo_lines.append(yolo_line)

                if drop_truncated and truncation_policy == "frame" and frame_has_truncated:
                    # strict mode: any border-cut object disqualifies the whole image
                    stats["frames_dropped_truncated"] += 1
                    continue
                if min_visibility > 0.0 and occlusion_policy == "frame" and frame_has_occluded:
                    # strict mode: any over-occluded object disqualifies the whole image
                    stats["frames_dropped_occluded"] += 1
                    continue
                if frame_has_degenerate:
                    # a degenerate box means an object is present but a label could
                    # not be written for it -> never keep an image with an
                    # unlabeled object, discard the whole frame
                    stats["frames_dropped_degenerate"] += 1
                    continue
                if (min_visibility > 0.0 or drop_truncated) and not yolo_lines:
                    # every object in this frame failed the filters
                    stats["frames_skipped_empty"] += 1
                    continue
                stats["boxes_kept"] += len(yolo_lines)
                stats["frames_kept"] += 1

                new_base_name = f"{exp_number}_{source_folder_name}_{camera_id}_rgb_{frame_number_str}"
                new_image_path = final_output_dir / f"{new_base_name}{img_ext}"
                new_label_path = final_output_dir / f"{new_base_name}.txt"

                shutil.copy2(image_path, new_image_path)
                with open(new_label_path, 'w') as f:
                    f.write("\n".join(yolo_lines))
                processed_count += 1
            except Exception as e:
                print(f"  - ? UNEXPECTED ERROR processing {npy_file.name}: {e}")
                traceback.print_exc()

    print("\n" + "=" * 60)
    if processed_count > 0:
        print(f"?? Conversion Complete! Processed a total of {processed_count} frames.")
    else:
        print("\n?? No frames were processed.")
    print("--- FILTERING SUMMARY ---")
    print(f"  > Bounding boxes kept              : {stats['boxes_kept']}")
    print(f"  > Dropped - occluded by objects    : {stats['boxes_dropped_occluded']}"
          + (f" (min_visibility={min_visibility})" if min_visibility > 0.0 else " (filter off)"))
    print(f"  > Dropped - out of camera frame    : {stats['boxes_dropped_truncated']}"
          + ("" if drop_truncated else " (filter off — use --drop_truncated)"))
    print(f"  > Dropped - degenerate (zero-area) : {stats['boxes_dropped_degenerate']}")
    print(f"  > Frames discarded (border-cut obj): {stats['frames_dropped_truncated']}"
          + ("" if (drop_truncated and truncation_policy == "frame") else " (policy 'box' — only labels removed, images kept)"))
    print(f"  > Frames discarded (occluded obj)  : {stats['frames_dropped_occluded']}"
          + ("" if (min_visibility > 0.0 and occlusion_policy == "frame") else " (policy 'box' — only labels removed, images kept)"))
    print(f"  > Frames discarded (degenerate box): {stats['frames_dropped_degenerate']}")
    print(f"  > Frames skipped (no box survived) : {stats['frames_skipped_empty']}")
    print("=" * 60)

    return dest_dir_final, stats

# ==========================================================
# FILTERING LOGIC
# ==========================================================

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    box1_x1, box1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    box1_x2, box1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    box2_x1, box2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    box2_x2, box2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    x_left, y_top = max(box1_x1, box2_x1), max(box1_y1, box2_y1)
    x_right, y_bottom = min(box1_x2, box2_x2), min(box1_y2, box2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = float(box1_area + box2_area - intersection_area)

    return intersection_area / union_area if union_area > 0 else 0.0

def filter_overlapping_boxes(yolo_dataset_path: Path, iou_threshold: float):
    """
    Scans a directory for images and labels, and removes pairs where bounding
    boxes overlap by a given threshold.
    """
    print("\n" + "="*60)
    print("?? STARTING OVERLAP FILTERING PROCESS")
    print(f"  > Target Directory: {yolo_dataset_path}")
    print(f"  > IoU Threshold: {iou_threshold}")
    print("="*60)
    
    image_extensions = ('.png', '.jpg', '.jpeg')
    deleted_count = 0

    if not yolo_dataset_path or not yolo_dataset_path.is_dir():
        print(f"? ERROR: Filtering path is not a valid directory: {yolo_dataset_path}")
        return

    for subdir, _, files in os.walk(yolo_dataset_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_path = Path(subdir) / file
                label_path = image_path.with_suffix('.txt')

                if label_path.exists():
                    with open(label_path, 'r') as f:
                        lines = f.readlines()

                    if len(lines) < 2:
                        continue

                    bboxes = [[float(p) for p in line.strip().split()[1:]] for line in lines if len(line.strip().split()) == 5]
                    
                    should_delete = False
                    for i in range(len(bboxes)):
                        for j in range(i + 1, len(bboxes)):
                            iou = calculate_iou(bboxes[i], bboxes[j])
                            if iou >= iou_threshold:
                                print(f"  - Deleting pair due to overlap >= {iou_threshold*100}% (IoU: {iou:.2f}) in {label_path.name}")
                                os.remove(image_path)
                                os.remove(label_path)
                                deleted_count += 1
                                should_delete = True
                                break
                        if should_delete:
                            break
                            
    print("\n" + "=" * 60)
    if deleted_count > 0:
        print(f"? Filtering Complete! Total files deleted: {deleted_count * 2} ({deleted_count} pairs)")
    else:
        print("? Filtering Complete! No overlapping boxes found.")
    print("=" * 60)
    return deleted_count