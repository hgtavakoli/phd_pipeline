import os
import yaml
import tempfile
import traceback
from glob import glob
from ultralytics import YOLO
from src.common import logging_utils, distributed_utils


def run_evaluation_pipeline(config):
    """
    Evaluates trained models, parsing experiment names to create structured logs.
    This is typically run as a single-process task on the master node.
    """
    if not distributed_utils.is_master_process():
        return

    # Configuration Paths and Parameters
    yolo_runs_eval_dir = config['evaluation']['source_yolo_runs_dir']
    custom_log_dir = config['paths']['custom_log_dir']
    class_names = config['classes']['names']
    num_classes = len(class_names)
    split_to_eval = config['evaluation']['split_to_eval_on']

    actual_data_path_for_eval = config['paths'].get(f"{split_to_eval}_data_path", config['paths']['test_data_path'])
    if not os.path.isdir(actual_data_path_for_eval):
        print(f"?? Error: Evaluation data directory not found: {actual_data_path_for_eval}")
        return

    # --- CSV Setup for all evaluation results ---
    eval_csv_path = os.path.join(custom_log_dir, "evaluation_summaries", "all_evaluation_results.csv")
    # <<< MODIFIED: Added 'Model_File' to the CSV header to track which .pt file is being evaluated >>>
    eval_csv_header = [
        "YOLO_Experiment_Name", "Model_File", "Source_Dataset", "Source_Exp_Name", "Data_Size", "Camera",
        "Split_Evaluated_On", "Evaluation_Data_Path",
        "Test_mAP50(B)_Mean", "Test_mAP50-95(B)_Mean", "Test_Precision(B)_Mean", "Test_Recall(B)_Mean"
    ]
    for cn in class_names:
        eval_csv_header.extend([f"Test_P_{cn}", f"Test_R_{cn}", f"Test_mAP50_{cn}", f"Test_mAP50-95_{cn}"])
    logging_utils.initialize_csv(eval_csv_path, eval_csv_header)

    # <<< MODIFIED: Changed glob pattern to find all .pt files, not just epoch100.pt >>>
    experiment_weight_paths = glob(os.path.join(yolo_runs_eval_dir, "*", "weights", "*.pt"))
    if not experiment_weight_paths:
        # <<< MODIFIED: Updated message to reflect the change in what is being searched for >>>
        print(f"No '.pt' weight files found in any subdirectories of {yolo_runs_eval_dir}")
        return

    print(f"Found {len(experiment_weight_paths)} trained models to evaluate.")

    for weights_path in experiment_weight_paths:
        # <<< MODIFIED: Extract the model filename (e.g., 'best.pt', 'epoch50.pt') >>>
        model_filename = os.path.basename(weights_path)
        yolo_experiment_run_name = os.path.basename(os.path.dirname(os.path.dirname(weights_path)))
        parts = yolo_experiment_run_name.split('_')
        # Set default values
        source_dataset, exp_name, size, camera = "unknown", yolo_experiment_run_name, "unknown", "unknown"

        # New, flexible parsing logic
        if len(parts) >= 2:
            # Assumes the last part is always the camera identifier
            camera = parts[-1]
            # Everything before the last part is the base experiment name
            exp_name = "_".join(parts[:-1])
        else:
            # This handles cases where the name has no underscores
            print(f"?? Warning: Could not parse camera from experiment name '{yolo_experiment_run_name}'.")
        print(f"\n{'=' * 20} Starting Evaluation {'=' * 20}")
        print(f"?? Evaluating: {yolo_experiment_run_name}")
        # <<< MODIFIED: Also print the specific model file being evaluated >>>
        print(f"   Weights: {model_filename}")
        print(f"   Data Split: '{split_to_eval}' at {actual_data_path_for_eval}")

        data_config_dict = {
            'train': actual_data_path_for_eval,
            'val': actual_data_path_for_eval,
            split_to_eval: actual_data_path_for_eval,
            'names': class_names
        }

        tmp_yaml_path = None
        try:
            fd, tmp_yaml_path = tempfile.mkstemp(suffix='.yaml', text=True)
            with os.fdopen(fd, 'w') as tmp_file:
                yaml.dump(data_config_dict, tmp_file)

            model = YOLO(weights_path)
            eval_args = {
                "data": tmp_yaml_path,
                "imgsz": config['evaluation']['img_size'],
                "batch": config['evaluation']['batch_size'],
                "split": split_to_eval,
                "project": os.path.join(custom_log_dir, "evaluation_yolo_outputs"),
                # <<< MODIFIED: Make the output name unique by including the model filename >>>
                "name": f"{yolo_experiment_run_name}_eval_on_{split_to_eval}_({model_filename.replace('.pt', '')})",
                "save_json": True,
            }
            eval_results_obj = model.val(**eval_args)

            test_precision_mean, test_recall_mean, test_mAP50_mean, test_mAP5095_mean = -1.0, -1.0, -1.0, -1.0
            per_class_p, per_class_r, per_class_ap50, per_class_ap = [], [], [], []

            if hasattr(eval_results_obj, 'box'):
                box_metrics = eval_results_obj.box
                test_precision_mean = round(box_metrics.mp, 4)
                test_recall_mean = round(box_metrics.mr, 4)
                test_mAP50_mean = round(box_metrics.map50, 4)
                test_mAP5095_mean = round(box_metrics.map, 4)

                per_class_p = box_metrics.p.tolist() if hasattr(box_metrics, 'p') else []
                per_class_r = box_metrics.r.tolist() if hasattr(box_metrics, 'r') else []
                if hasattr(box_metrics, 'all_ap'):
                    per_class_ap50 = box_metrics.all_ap[:, 0].tolist()
                if hasattr(box_metrics, 'maps'):
                    per_class_ap = box_metrics.maps.tolist()

            log_row_values_for_csv = [
                yolo_experiment_run_name,
                # <<< MODIFIED: Added the model filename to the CSV row data >>>
                model_filename,
                source_dataset, exp_name, size, camera,
                split_to_eval, actual_data_path_for_eval,
                test_mAP50_mean, test_mAP5095_mean, test_precision_mean, test_recall_mean
            ]

            for i in range(num_classes):
                log_row_values_for_csv.append(per_class_p[i] if i < len(per_class_p) else -1)
                log_row_values_for_csv.append(per_class_r[i] if i < len(per_class_r) else -1)
                log_row_values_for_csv.append(per_class_ap50[i] if i < len(per_class_ap50) else -1)
                log_row_values_for_csv.append(per_class_ap[i] if i < len(per_class_ap) else -1)

            logging_utils.append_to_csv(eval_csv_path, log_row_values_for_csv)
            print(f"? Evaluation results logged to {eval_csv_path}")

            # <<< MODIFIED: Made the text summary path unique for each model evaluated >>>
            txt_summary_path = os.path.join(
                custom_log_dir, "evaluation_summaries", yolo_experiment_run_name,
                f"eval_summary_on_{split_to_eval}_({model_filename.replace('.pt', '')}).txt"
            )
            summary_content_for_txt = {
                "YOLO Experiment Name": yolo_experiment_run_name,
                # <<< MODIFIED: Added Model File to the text summary >>>
                "Model File": model_filename,
                "Source Dataset": source_dataset,
                "Source Experiment": exp_name,
                "Data Size": size,
                "Camera": camera,
                "Weights Path": weights_path,
                "Evaluated on Split": split_to_eval,
                "Evaluation Data Path": actual_data_path_for_eval,
                "Image Size": config['evaluation']['img_size'],
                "Test mAP50(B) (Mean)": f"{test_mAP50_mean:.4f}",
                "Test mAP50-95(B) (Mean)": f"{test_mAP5095_mean:.4f}",
                "Test Precision(B) (Mean)": f"{test_precision_mean:.4f}",
                "Test Recall(B) (Mean)": f"{test_recall_mean:.4f}",
                "\n--- Per-Class Evaluation Metrics ---": ""
            }
            if not per_class_ap50 and not per_class_ap:
                summary_content_for_txt["Per-Class Metrics Note"] = "Per-class metrics not generated by model."
            else:
                for i, name in enumerate(class_names):
                    p_cls = per_class_p[i] if i < len(per_class_p) else -1
                    r_cls = per_class_r[i] if i < len(per_class_r) else -1
                    ap50_cls = per_class_ap50[i] if i < len(per_class_ap50) else -1
                    ap_cls = per_class_ap[i] if i < len(per_class_ap) else -1
                    summary_content_for_txt[f"Class: {name} - Test Precision"] = f"{p_cls:.4f}"
                    summary_content_for_txt[f"Class: {name} - Test Recall"] = f"{r_cls:.4f}"
                    summary_content_for_txt[f"Class: {name} - Test mAP@0.5"] = f"{ap50_cls:.4f}"
                    summary_content_for_txt[f"Class: {name} - Test mAP@0.5:0.95"] = f"{ap_cls:.4f}"

            logging_utils.save_text_summary(txt_summary_path, summary_content_for_txt)
            print(f"? Text summary for evaluation saved to {txt_summary_path}")

        except Exception as e:
            print(f"?? Error during evaluation for {yolo_experiment_run_name} ({model_filename}):")
            traceback.print_exc()
        finally:
            if tmp_yaml_path and os.path.exists(tmp_yaml_path):
                os.remove(tmp_yaml_path)

    print("\n?? All evaluations processed.")