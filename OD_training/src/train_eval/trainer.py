import os
import yaml
import tempfile
import traceback
from ultralytics import YOLO
from src.common import logging_utils

def run_training_pipeline(config):
    """
    Manages the training loop for multiple datasets, configurations, and cameras.
    """

    base_data_dir = config['paths']['base_data_dir']
    val_data_path = config['paths']['val_data_path']
    yolo_project_dir = config['paths']['yolo_project_dir']
    custom_log_dir = config['paths']['custom_log_dir']

    class_names = config['classes']['names']
    num_classes = len(class_names)

    # --- CSV Setup ---
    training_csv_path = os.path.join(custom_log_dir, "training_summaries", "all_training_runs.csv")
    training_csv_header = [
        "YOLO_Experiment_Name", "Source_Dataset", "Source_Exp_Name", "Data_Size", "Camera",
        "Epochs", "Batch_Size", "Img_Size",
        "Train_mAP50(B)", "Train_mAP50-95(B)", "Train_Precision(B)", "Train_Recall(B)",
        "Val_mAP50(B)", "Val_mAP50-95(B)", "Val_Precision(B)", "Val_Recall(B)"
    ]
    for cn in class_names:
        training_csv_header.extend([f"Val_P_{cn}", f"Val_R_{cn}", f"Val_mAP50_{cn}", f"Val_mAP_{cn}"])
    logging_utils.initialize_csv(training_csv_path, training_csv_header)

    for dataset_group in config['training']['datasets_to_train']:
        for exp_name_segment in config['training']['exp_names_to_train']:
            for size_segment in config['training']['sizes_to_train']:
                for camera_segment in config['training'].get('cameras_to_train', ['']):
                    
                    current_train_data_path = os.path.join(
                        base_data_dir, dataset_group, exp_name_segment, size_segment, camera_segment
                    )
                    yolo_experiment_run_name = f"{dataset_group}_{exp_name_segment}_{size_segment}_{camera_segment}"

                    if not os.path.exists(current_train_data_path):
                        print(f"? Skipping {current_train_data_path}: Path does not exist")
                        continue

                    print(f"\n{'=' * 20} Starting Training {'=' * 20}")
                    print(f"?? Experiment: {yolo_experiment_run_name}")
                    print(f"   Train Data: {current_train_data_path}")
                    print(f"   Val Data: {val_data_path}")

                    data_config_dict = {
                        'train': current_train_data_path,
                        'val': val_data_path,
                        'nc': num_classes,
                        'names': class_names
                    }

                    tmp_yaml_path = None
                    try:
                        fd, tmp_yaml_path = tempfile.mkstemp(suffix='.yaml', text=True)
                        with os.fdopen(fd, 'w') as tmp_file:
                            yaml.dump(data_config_dict, tmp_file)

                        # Train
                        model = YOLO(config['training']['weights'])
                        train_args = {
                            "data": tmp_yaml_path,
                            "epochs": config['training']['epochs'],
                            "imgsz": config['training']['img_size'],
                            "batch": config['training']['batch_size'],
                            "name": yolo_experiment_run_name,
                            "project": yolo_project_dir,
                            "save": True,
                            "save_period": config['training'].get('save_period', -1),
                            "workers": config['training'].get('workers', 8),
                            "device": 0
                        }
                        train_args = {k: v for k, v in train_args.items() if v is not None}
                        train_results = model.train(**train_args)

                        print("? Training completed.")
                        final_train_metrics = train_results.metrics if hasattr(train_results, 'metrics') else {}

                        # Validation
                        print("? Running validation on the final model...")
                        val_results_obj = model.val(
                            data=tmp_yaml_path,
                            imgsz=config['training']['img_size'],
                            batch=config['training']['batch_size'],
                            project=yolo_project_dir,
                            name=f"{yolo_experiment_run_name}_val"
                        )

                        # Metrics extraction
                        val_precision_mean, val_recall_mean, val_mAP50_mean, val_mAP5095_mean = -1, -1, -1, -1
                        per_class_p, per_class_r, per_class_ap50, per_class_ap = [], [], [], []

                        if hasattr(val_results_obj, 'box'):
                            box_metrics = val_results_obj.box
                            val_precision_mean = box_metrics.mp
                            val_recall_mean = box_metrics.mr
                            val_mAP50_mean = box_metrics.map50
                            val_mAP5095_mean = box_metrics.map
                            per_class_p = getattr(box_metrics, 'p', []).tolist()
                            per_class_r = getattr(box_metrics, 'r', []).tolist()
                            if hasattr(box_metrics, 'all_ap'):
                                per_class_ap50 = box_metrics.all_ap[:, 0].tolist()
                            if hasattr(box_metrics, 'maps'):
                                per_class_ap = box_metrics.maps.tolist()

                        # Log CSV
                        log_row_values_for_csv = [
                            yolo_experiment_run_name, dataset_group, exp_name_segment, size_segment, camera_segment,
                            config['training']['epochs'], config['training']['batch_size'],
                            config['training']['img_size'],
                            round(final_train_metrics.get('metrics/mAP50(B)', -1), 4),
                            round(final_train_metrics.get('metrics/mAP50-95(B)', -1), 4),
                            round(final_train_metrics.get('metrics/precision(B)', -1), 4),
                            round(final_train_metrics.get('metrics/recall(B)', -1), 4),
                            val_mAP50_mean, val_mAP5095_mean, val_precision_mean, val_recall_mean
                        ]
                        for i in range(len(class_names)):
                            log_row_values_for_csv.append(per_class_p[i] if i < len(per_class_p) else -1)
                            log_row_values_for_csv.append(per_class_r[i] if i < len(per_class_r) else -1)
                            log_row_values_for_csv.append(per_class_ap50[i] if i < len(per_class_ap50) else -1)
                            log_row_values_for_csv.append(per_class_ap[i] if i < len(per_class_ap) else -1)

                        logging_utils.append_to_csv(training_csv_path, log_row_values_for_csv)
                        print(f"?? CSV results logged to {training_csv_path}")

                        # Save TXT Summary
                        txt_summary_path = os.path.join(
                            custom_log_dir, "training_summaries", yolo_experiment_run_name, "summary.txt"
                        )
                        summary_content_for_txt = {
                            "YOLO Experiment Name": yolo_experiment_run_name,
                            "Source Dataset": dataset_group,
                            "Source Experiment Segment": exp_name_segment,
                            "Data Size Segment": size_segment,
                            "Camera": camera_segment,
                            "Training Data Path": current_train_data_path,
                            "Validation Data Path": val_data_path,
                            "Epochs": config['training']['epochs'],
                            "Batch Size": config['training']['batch_size'],
                            "Image Size": config['training']['img_size'],
                            "Train mAP50(B)": f"{final_train_metrics.get('metrics/mAP50(B)', -1):.4f}",
                            "Train mAP50-95(B)": f"{final_train_metrics.get('metrics/mAP50-95(B)', -1):.4f}",
                            "Val mAP50(B) (Mean)": f"{val_mAP50_mean:.4f}",
                            "Val mAP50-95(B) (Mean)": f"{val_mAP5095_mean:.4f}",
                            "Val Precision(B) (Mean)": f"{val_precision_mean:.4f}",
                            "Val Recall(B) (Mean)": f"{val_recall_mean:.4f}",
                            "YOLO Run Directory": os.path.join(yolo_project_dir, yolo_experiment_run_name),
                            "\n--- Per-Class Validation Metrics ---": ""
                        }

                        if not per_class_ap50 and not per_class_ap:
                            summary_content_for_txt["Per-Class Metrics Note"] = "Per-class metrics not generated by model."
                        else:
                            for i, name in enumerate(class_names):
                                summary_content_for_txt[f"Class: {name} - Val Precision"] = f"{per_class_p[i]:.4f}" if i < len(per_class_p) else "-1"
                                summary_content_for_txt[f"Class: {name} - Val Recall"] = f"{per_class_r[i]:.4f}" if i < len(per_class_r) else "-1"
                                summary_content_for_txt[f"Class: {name} - Val mAP@0.5"] = f"{per_class_ap50[i]:.4f}" if i < len(per_class_ap50) else "-1"
                                summary_content_for_txt[f"Class: {name} - Val mAP@0.5:0.95"] = f"{per_class_ap[i]:.4f}" if i < len(per_class_ap) else "-1"

                        logging_utils.save_text_summary(txt_summary_path, summary_content_for_txt)
                        print(f"?? Text summary saved to {txt_summary_path}")

                    except Exception as e:
                        print(f"? Error during training/validation for {yolo_experiment_run_name}:")
                        traceback.print_exc()

                    finally:
                        if tmp_yaml_path and os.path.exists(tmp_yaml_path):
                            os.remove(tmp_yaml_path)
