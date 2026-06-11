# cl_src/cl_orchestrator.py
import os
import sys
import tempfile
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add parent directory to path to find the OD_training module
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from cl_src.data_utils import prepare_task_dataset, prepare_task_validation_set, BalancedBatchSampler
from ultralytics.data.build import build_yolo_dataset


def run_incremental_step(args):
    """Executes a single incremental training step with balanced batching."""
    # --- (Code from Step 1, 2, and 3 is exactly the same) ---
    class_offset = len(args.current_classes)
    all_class_names = args.current_classes + args.new_classes
    model_weights_path = Path(args.model_weights)
    if not model_weights_path.is_file():
        raise FileNotFoundError(f"Pretrained model weights not found at: {model_weights_path}")
    print(f"  - ✅ Pretrained model found at: {model_weights_path}")
    print("  - Initializing temporary YOLO model to extract configuration...")
    temp_model = YOLO(args.model_weights)
    model_cfg = get_cfg()
    for key, value in temp_model.args.items():
        setattr(model_cfg, key, value)
    print("  - ✅ Full configuration created.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        training_data_dir, new_fnames, replay_fnames = prepare_task_dataset(
            new_data_dir=Path(args.new_data_dir),
            replay_data_dir=Path(args.replay_data_dir),
            output_dir=tmp_path / "train",
            class_offset=class_offset
        )
        val_data_path_str = ''
        if args.val_new_data_dir and args.val_replay_data_dir:
            validation_data_dir = prepare_task_validation_set(
                new_val_dir=Path(args.val_new_data_dir),
                replay_val_dir=Path(args.val_replay_data_dir),
                output_dir=tmp_path / "val",
                class_offset=class_offset
            )
            val_data_path_str = str(validation_data_dir)
        else:
            print("  - INFO: Validation paths not provided. Validation will be skipped.")
        print("  - Building custom balanced dataloaloader...")
        yolo_dataset = build_yolo_dataset(
            cfg=model_cfg,
            img_path=str(training_data_dir),
            batch=args.batch_size,
            data={'names': all_class_names, 'channels': 3},
            mode="train"
        )
        fname_to_idx = {Path(f).name: i for i, f in enumerate(yolo_dataset.im_files)}
        new_indices = [fname_to_idx[fname] for fname in new_fnames if fname in fname_to_idx]
        replay_indices = [fname_to_idx[fname] for fname in replay_fnames if fname in fname_to_idx]

        balanced_sampler = BalancedBatchSampler(
            dataset=yolo_dataset,
            new_indices=new_indices,
            replay_indices=replay_indices,
            batch_size=args.batch_size,
            replay_ratio=args.replay_ratio
        )

        # ================================================================= #
        # == NEW: Added a detailed summary of the batching strategy      == #
        # ================================================================= #
        print("\n" + "="*25 + " Batch Sampler Info " + "="*25)
        print(f"  - Total unique NEW images found:    {len(new_indices)}")
        print(f"  - Total unique REPLAY images found: {len(replay_indices)}")
        print(f"  - Requested Replay Ratio:           {args.replay_ratio:.2f} ({args.replay_ratio * 100:.0f}%)")
        print("-" * 72)
        print(f"  - Batch Composition:")
        print(f"    - Total Batch Size:               {balanced_sampler.batch_size}")
        print(f"    - New Samples per Batch:          {balanced_sampler.new_samples_per_batch}")
        print(f"    - Replay Samples per Batch:       {balanced_sampler.replay_samples_per_batch}")
        print("-" * 72)

        num_batches_per_epoch = len(balanced_sampler)
        total_images_per_epoch = num_batches_per_epoch * balanced_sampler.batch_size
        print(f"  - Total Batches per Epoch:          {num_batches_per_epoch}")
        print(f"  - Total Images processed per Epoch: {total_images_per_epoch} (approx.)")
        print("="*72)


        custom_dataloader = DataLoader(
            yolo_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            sampler=None,
            collate_fn=getattr(yolo_dataset, 'collate_fn', None),
            batch_sampler=balanced_sampler
        )

        # --- 4. Create the data.yaml file for validation and metadata ---
        data_yaml_content = {
            'train': str(training_data_dir),
            'val': val_data_path_str,
            'nc': len(all_class_names),
            'names': all_class_names
        }
        tmp_yaml_path = tmp_path / 'data.yaml'
        with open(tmp_yaml_path, 'w') as f:
            yaml.dump(data_yaml_content, f, default_flow_style=False)

        # --- 5. Initialize the final model and call .train() ---
        print("\n🚀 Initializing model for training...")
        model = YOLO(args.model_weights)

        def on_pretrain_routine_start_callback(trainer):
            """This function will be called by YOLO, replacing its loader with ours."""
            print("  - Callback triggered: Replacing default train_loader with custom balanced loader.")
            trainer.train_loader = custom_dataloader

        model.add_callback("on_pretrain_routine_start", on_pretrain_routine_start_callback)

        print("   Starting YOLO training...")

        model.train(
            data=str(tmp_yaml_path),
            project=args.output_dir,
            name=args.exp_name,
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch_size
        )
        print(f"✅ Training finished.")

        # --- 6. Report the Final Model Path ---
        final_model_path = Path(args.output_dir) / args.exp_name / "weights" / "best.pt"
        if not final_model_path.exists():
            print(f"\n⚠️ WARNING: Could not find the final 'best.pt' model in the primary expected path.")
            print(f"   Please check the directory '{Path(args.output_dir) / args.exp_name}' for outputs.")
        else:
            print(f"\n🎉 New model successfully trained and saved at:\n   {final_model_path.resolve()}")