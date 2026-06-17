import os
import yaml
import tempfile
import traceback
from pathlib import Path
from ultralytics import YOLO
import torch.distributed as dist

from ..common import logging_utils, distributed_utils


def _clear_stale_label_caches(*paths):
    """Delete ultralytics label .cache files for the given data dirs.

    The cache stores ABSOLUTE image paths, so a cache written under Docker
    (/workspace/...) — or on another machine — makes a local run fail to find the
    images ('no labels in cache') even though the images+labels are right there.
    Rebuilding the cache is cheap; clearing stale ones is the robust fix for
    switching between Docker and native runs on the same dataset folder."""
    for p in paths:
        if not p:
            continue
        p = Path(p)
        candidates = [p.with_suffix('.cache')]          # ultralytics: <imgdir>.cache (sibling)
        if p.is_dir():
            candidates += list(p.glob('*.cache'))
        for c in candidates:
            try:
                if c.is_file():
                    c.unlink()
                    print(f"  - cleared stale label cache: {c}")
            except Exception:
                pass


def run_periodic_evaluation(weights_path, epoch, config, yolo_experiment_run_name, custom_log_dir, class_names):
    """
    Runs an evaluation cycle on the TEST SET for a given checkpoint and logs results.
    This function should only be called by the master process.
    """
    print(f"\n{'-' * 20} Running Periodic Evaluation on TEST SET for Epoch {epoch} {'-' * 20}")
    print(f"   Weights: {os.path.basename(weights_path)}")

    eval_log_subdir = os.path.join(custom_log_dir, "periodic_test_evaluations")
    eval_csv_path = os.path.join(eval_log_subdir, "evaluation_summary.csv")
    yolo_output_dir = os.path.join(eval_log_subdir, "yolo_native_outputs")
    os.makedirs(yolo_output_dir, exist_ok=True)

    eval_csv_header = [
        "YOLO_Experiment_Name", "Epoch", "Model_File",
        "Test_mAP50(B)_Mean", "Test_mAP50-95(B)_Mean", "Test_Precision(B)_Mean", "Test_Recall(B)_Mean"
    ]
    num_classes = len(class_names)
    for cn in class_names:
        eval_csv_header.extend([f"Test_P_{cn}", f"Test_R_{cn}", f"Test_mAP50_{cn}", f"Test_mAP50-95_{cn}"])
    logging_utils.initialize_csv(eval_csv_path, eval_csv_header)

    test_data_path = config['paths']['test_data_path']
    if not os.path.isdir(test_data_path):
        print(f"?? Error: Test data directory not found: {test_data_path}")
        return

    data_config_dict = {
        'test': test_data_path,
        'names': class_names
    }

    tmp_yaml_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(data_config_dict, tmp_file)
            tmp_yaml_path = tmp_file.name

        model = YOLO(weights_path)
        val_results_obj = model.val(
            data=tmp_yaml_path,
            split='test',
            imgsz=config['evaluation']['img_size'],
            batch=config['evaluation']['batch_size'],
            project=yolo_output_dir,
            name=f"eval_epoch_{epoch}",
            save_json=True
        )

        test_mAP50_mean = round(val_results_obj.box.map50, 4)
        test_mAP5095_mean = round(val_results_obj.box.map, 4)
        test_precision_mean = round(val_results_obj.box.mp, 4)
        test_recall_mean = round(val_results_obj.box.mr, 4)

        per_class_p = val_results_obj.box.p.tolist()
        per_class_r = val_results_obj.box.r.tolist()
        per_class_ap50 = val_results_obj.box.all_ap[:, 0].tolist()
        per_class_ap = val_results_obj.box.maps.tolist()

        log_row = [
            yolo_experiment_run_name, epoch, os.path.basename(weights_path),
            test_mAP50_mean, test_mAP5095_mean, test_precision_mean, test_recall_mean
        ]

        for i in range(num_classes):
            log_row.append(per_class_p[i] if i < len(per_class_p) else -1)
            log_row.append(per_class_r[i] if i < len(per_class_r) else -1)
            log_row.append(per_class_ap50[i] if i < len(per_class_ap50) else -1)
            log_row.append(per_class_ap[i] if i < len(per_class_ap) else -1)

        logging_utils.append_to_csv(eval_csv_path, log_row)
        print(f"? Periodic test evaluation results logged to {eval_csv_path}")

    except Exception as e:
        print(f"?? Error during periodic evaluation for epoch {epoch}:")
        traceback.print_exc()
    finally:
        if tmp_yaml_path and os.path.exists(tmp_yaml_path):
            os.remove(tmp_yaml_path)


def run_training_pipeline(config):
    """
    Manages the full training lifecycle.
    """
    paths_config = config['paths']
    training_config = config['training']
    class_names = config['classes']['names']
    num_classes = len(class_names)

    base_data_dir = paths_config['base_data_dir']
    val_data_path = paths_config['val_data_path']
    yolo_project_dir = paths_config['yolo_project_dir']
    custom_log_dir = paths_config['custom_log_dir']

    def to_list(value):
        return [value] if isinstance(value, str) else value if isinstance(value, list) else []

    base_exp_name = to_list(training_config.get('exp_names_to_train', []))[0]
    cameras_to_loop = to_list(training_config.get('cameras_to_train', []))

    if distributed_utils.is_master_process():
        print("--- Starting Training Pipeline ---")
        print(f"  - Base Experiment Name: {base_exp_name}")
        print(f"  - Cameras to Train On: {cameras_to_loop}")
        print(f"  - Project Directory: {yolo_project_dir}")
        print("-" * 60)

    for camera_segment in cameras_to_loop:
        yolo_experiment_run_name = f"{base_exp_name}_{camera_segment}"
        current_train_data_path = Path(base_data_dir) / camera_segment

        if not current_train_data_path.is_dir():
            if distributed_utils.is_master_process():
                print(f"⚠️ Skipping {yolo_experiment_run_name}: Path not found at '{current_train_data_path}'")
            continue

        if distributed_utils.is_master_process():
            print(f"\n{'=' * 25} Starting Run: {yolo_experiment_run_name} {'=' * 25}")
            print(f"  - Training Data: {current_train_data_path}")

        data_yaml_content = {
            'train': str(current_train_data_path),
            'val': val_data_path,
            'test': paths_config['test_data_path'],
            'nc': num_classes,
            'names': class_names
        }

        tmp_yaml_path = None
        try:
            # Only the master process creates the temp file.
            if distributed_utils.is_master_process():
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
                    yaml.dump(data_yaml_content, tmp_file, default_flow_style=False)
                    tmp_yaml_path = tmp_file.name

            # **FIX**: Synchronize all processes to ensure the file is created before proceeding.
            if distributed_utils.get_world_size() > 1:
                dist.barrier()
                # Broadcast the file path from the master process to all other processes.
                tmp_path_list = [tmp_yaml_path]
                dist.broadcast_object_list(tmp_path_list, src=0)
                tmp_yaml_path = tmp_path_list[0]

            model = YOLO(training_config['weights'])
            save_period = training_config.get('save_period', -1)

            def on_model_save_callback(trainer):
                # CRITICAL: Only the master process should run evaluation and log.
                if distributed_utils.is_master_process() and os.path.basename(trainer.last).startswith('epoch'):
                    run_periodic_evaluation(
                        weights_path=trainer.last,
                        epoch=trainer.epoch,
                        config=config,
                        yolo_experiment_run_name=yolo_experiment_run_name,
                        custom_log_dir=os.path.join(custom_log_dir, camera_segment),
                        class_names=class_names
                    )

            model.add_callback("on_model_save", on_model_save_callback)

            # Drop any label cache left by a different environment (Docker stores
            # /workspace/... paths) so ultralytics rescans against the real local
            # paths instead of failing to find the images.
            if distributed_utils.is_master_process():
                _clear_stale_label_caches(current_train_data_path, val_data_path,
                                          paths_config.get('test_data_path'))

            # **FIX**: The 'rank' argument is removed. Ultralytics handles it automatically.
            model.train(
                data=tmp_yaml_path,
                project=yolo_project_dir,
                # Run subfolder = camera only; the project dir already carries the
                # (long) dataset name. Repeating it overflowed Windows MAX_PATH
                # (WinError 206) for multi-class dataset names.
                name=camera_segment,
                epochs=training_config['epochs'],
                imgsz=training_config['img_size'],
                batch=training_config['batch_size'],
                save_period=save_period
            )

        except Exception as e:
            if distributed_utils.is_master_process():
                print(f"❌ ERROR during training run '{yolo_experiment_run_name}':")
                traceback.print_exc()
        finally:
            if distributed_utils.is_master_process() and tmp_yaml_path and os.path.exists(tmp_yaml_path):
                os.remove(tmp_yaml_path)
            # Ensure all processes wait until the master has cleaned up before continuing.
            if distributed_utils.get_world_size() > 1:
                dist.barrier()

    if distributed_utils.is_master_process():
        print(f"\n{'=' * 25} All Training Runs Completed {'=' * 25}")