"""
SmartSDG — unified synthetic data generation entry point.

Runs the full per-dataset pipeline inside NVIDIA Isaac Sim:

    1. GENERATE   Replicator renders randomized scenes (src/generation.py)
    2. CONVERT    raw output -> YOLO dataset with quality filters
                  (src/yolo_converter.py, enable with --convert_to_yolo)
    3. FILTER     optional whole-frame overlap filter (--filter_overlaps)
    4. REPORT     FINAL DATASET SUMMARY: every kept/discarded frame accounted for

Must be executed with Isaac Sim's bundled Python, e.g.:

    C:\\Users\\hota02\\Documents\\omniverse\\python.bat main.py ^
        --config "src\\config\\config.yaml" ^
        --multi_object --components Cabin,GlueGun,BLid ^
        --light_mode Medium --num_samples 100 --load_env plane ^
        --width 640 --height 480 ^
        --convert_to_yolo --filter_overlaps --drop_truncated

All hyperparameters are documented in docs/HYPERPARAMETERS.md. Validated
defaults are baked into the argparse definitions below and into
src/config/config.yaml — change them deliberately and record changes in
CHANGELOG.md.
"""

import argparse
import random
import sys
from dataclasses import asdict
from pathlib import Path

# Make the project root importable regardless of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config_process import load_and_process_config
from src.generation import run_generation_task
from src.yolo_converter import run_yolo_conversion, filter_overlapping_boxes

# Isaac Sim 4.5 still ships the deprecated omni.isaac.kit alias; prefer it for
# compatibility and fall back to the new module path.
try:
    from omni.isaac.kit import SimulationApp
except ImportError:
    print("Could not import SimulationApp, trying from isaacsim...", file=sys.stderr)
    try:
        from isaacsim.simulation_app import SimulationApp
    except ImportError:
        print("CRITICAL ERROR: Could not import SimulationApp.", file=sys.stderr)
        sys.exit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI definition. Defaults reflect the validated baseline (2026-06-10)."""
    parser = argparse.ArgumentParser(
        description="Unified SDG script: Isaac Sim generation + YOLO conversion + quality filtering.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- Core run definition -------------------------------------------------
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "src/config/config.yaml"),
                        help="Path to the YAML configuration file.")
    parser.add_argument("--light_mode", type=str, required=True, choices=["Low", "Medium", "High"],
                        help="Lighting intensity band (see LIGHT_INTENSITY_MAP in config.yaml).")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Frames to render per camera (2 cameras => 2x this many raw frames).")
    parser.add_argument("--load_env", type=str, default="false", choices=["true", "plane", "false"],
                        help="Background: 'true' = full 3D scene, 'plane' = flat ground, 'false' = void.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for Python/NumPy RNGs (reproducible object/camera placement).")

    # --- Object selection -----------------------------------------------------
    parser.add_argument("--multi_object", action="store_true",
                        help="Scatter multiple components per scene instead of a single one.")
    parser.add_argument("--components", type=str, default=None,
                        help="Multi-object mode: comma-separated subset, e.g. 'GlueGun,Cabin,Ylid'.\n"
                             "Class IDs stay global. Default: all components in COMPONENT_USD_MAP.")
    parser.add_argument("--component_class", type=str, default=None,
                        help="Single-object mode: which component to render.")

    # --- Scene placement (Group 1 hyperparameters) ----------------------------
    parser.add_argument("--spawn_radius", type=float, default=None,
                        help="Multi-object spawn radius in metres (overrides config multi_spawn_radius).\n"
                             "Smaller = objects nearer the scene center = fewer border cuts,\n"
                             "object size on screen unchanged.")
    parser.add_argument("--min_separation", type=float, default=0.05,
                        help="Multi-object: minimum XY distance (m) between object centers\n"
                             "(collision-aware placement, prevents occlusion). 0 disables.")

    # --- Renderer / ray-casting quality (Group 3 hyperparameters) -------------
    # Defaults come from config.yaml SIMULATION_CONFIG; these flags override per run.
    parser.add_argument("--renderer", type=str, default=None, choices=["PathTracing", "RayTracedLighting"],
                        help="Render backend. PathTracing = accurate (default), RayTracedLighting = fast.")
    parser.add_argument("--spp", type=int, default=None,
                        help="Samples (rays) per pixel per frame — main quality/speed knob.\n"
                             "8 = fast iteration (denoiser compensates), 32-64 = final datasets.")
    parser.add_argument("--max_bounces", type=int, default=None,
                        help="Maximum light-path bounces.")
    parser.add_argument("--denoiser", type=str, default=None, choices=["on", "off"],
                        help="OptiX AI denoiser. Keep 'on' when using low --spp.")

    # --- Output resolution ----------------------------------------------------
    parser.add_argument("--width", type=int, default=None,
                        help="Image width in px (overrides config camera img_width).")
    parser.add_argument("--height", type=int, default=None,
                        help="Image height in px (overrides config camera img_height).")

    # --- YOLO conversion & quality filters (Group 2 hyperparameters) ----------
    # INVARIANT: with the default 'frame' policies, no kept image ever contains
    # an unlabeled object — failing objects discard the whole frame.
    parser.add_argument("--convert_to_yolo", action="store_true",
                        help="Convert raw Replicator output to a YOLO dataset after generation.")
    parser.add_argument("--min_visibility", type=float, default=0.95,
                        help="Keep an object only if at least this fraction is visible\n"
                             "(occlusion BY OTHER OBJECTS, Replicator occlusionRatio). 0 disables.")
    parser.add_argument("--occlusion_policy", type=str, default="frame", choices=["box", "frame"],
                        help="'frame' (default): discard the whole image when any object is too occluded.\n"
                             "'box': remove only the label — AVOID, leaves unlabeled pixels.")
    parser.add_argument("--drop_truncated", action="store_true",
                        help="Treat objects touching the image border (partially outside the\n"
                             "camera view) as invalid.")
    parser.add_argument("--truncation_policy", type=str, default="frame", choices=["box", "frame"],
                        help="'frame' (default): discard the whole image when any object is border-cut.\n"
                             "'box': remove only the label — AVOID, leaves unlabeled pixels.")
    parser.add_argument("--filter_overlaps", action="store_true",
                        help="Delete image/label pairs whose label boxes overlap >= --iou_threshold.")
    parser.add_argument("--iou_threshold", type=float, default=None,
                        help="IoU threshold for --filter_overlaps. Default: YOLO_FILTERING.IOU_THRESHOLD\n"
                             "from config.yaml (0.7).")
    return parser


def apply_cli_overrides(args: argparse.Namespace, config) -> None:
    """Fold CLI overrides into the loaded config (spawn box, resolution)."""
    if args.spawn_radius is not None:
        scene = config.generation_params.scene
        spawn = config.generation_params.spawning
        spawn.multi_spawn_radius = args.spawn_radius
        config.MULTI_SPAWN_AREA_MIN = [scene.center_x - spawn.multi_spawn_radius,
                                       scene.center_y - spawn.multi_spawn_radius,
                                       scene.truck_z_position]
        config.MULTI_SPAWN_AREA_MAX = [scene.center_x + spawn.multi_spawn_radius,
                                       scene.center_y + spawn.multi_spawn_radius,
                                       scene.truck_z_position + spawn.multi_spawn_z_variation]
        print(f"Spawn radius override: {args.spawn_radius} m")

    # Resolution must update BOTH the camera render products and the simulation
    # viewport so the YOLO labels always match the output images.
    if args.width is not None:
        config.generation_params.camera.img_width = args.width
        config.SIMULATION_CONFIG.width = args.width
    if args.height is not None:
        config.generation_params.camera.img_height = args.height
        config.SIMULATION_CONFIG.height = args.height
    print(f"Render resolution: {config.generation_params.camera.img_width}"
          f"x{config.generation_params.camera.img_height}")

    sim = config.SIMULATION_CONFIG
    if args.renderer:
        sim.renderer = args.renderer
    if args.spp:
        sim.samples_per_pixel_per_frame = args.spp
    if args.max_bounces:
        sim.max_bounces = args.max_bounces
    if args.denoiser is not None:
        sim.denoiser = sim.use_denoiser = (args.denoiser == "on")
    print(f"Renderer: {sim.renderer} | spp: {sim.samples_per_pixel_per_frame} | "
          f"max_bounces: {sim.max_bounces} | denoiser: {'on' if sim.denoiser else 'off'}")


def validate_component_selection(args: argparse.Namespace, config) -> None:
    """Resolve and validate --component_class / --components against the config."""
    component_choices = list(config.COMPONENT_USD_MAP.keys())

    if not args.multi_object and not args.component_class:
        args.component_class = component_choices[0]
    if args.component_class and args.component_class not in component_choices:
        raise ValueError(f"Invalid --component_class '{args.component_class}'. "
                         f"Valid choices are : {component_choices}")

    args.components_list = None
    if args.components:
        args.components_list = [c.strip() for c in args.components.split(",") if c.strip()]
        invalid = [c for c in args.components_list if c not in component_choices]
        if invalid:
            raise ValueError(f"Invalid --components entries {invalid}. "
                             f"Valid choices are : {component_choices}")

    if args.multi_object:
        subset = args.components_list or component_choices
        print(f"Mode: MULTI-object ({', '.join(subset)})")
    else:
        print(f"Mode: SINGLE-object ({args.component_class})")


def print_final_summary(args, conv_stats, overlap_deleted, iou_threshold, yolo_output_dir) -> None:
    """Account for every frame: rendered, kept, and each discard reason."""
    final_images = sum(1 for _ in Path(yolo_output_dir).rglob("*.png")) if yolo_output_dir else 0
    s = conv_stats
    print("\n" + "#" * 64)
    print("##              FINAL DATASET SUMMARY                        ##")
    print("#" * 64)
    print(f"  Requested samples                  : {args.num_samples}  (x2 cameras = {args.num_samples * 2} frames)")
    print(f"  Frames rendered by Isaac Sim       : {s['frames_total']}")
    print(f"  Frames discarded - object cut by image border : {s['frames_dropped_truncated']}")
    print(f"  Frames discarded - object occluded < {args.min_visibility*100:.0f}% visible : {s['frames_dropped_occluded']}")
    print(f"  Frames discarded - degenerate (unlabelable) box : {s['frames_dropped_degenerate']}")
    print(f"  Frames discarded - no labels survived filters : {s['frames_skipped_empty']}")
    if overlap_deleted is not None:
        print(f"  Frames deleted   - boxes overlapping IoU >= {iou_threshold} : {overlap_deleted}")
    else:
        print(f"  Frames deleted   - overlap filter not enabled (--filter_overlaps)")
    print(f"  ------------------------------------------------------------")
    print(f"  FINAL images in dataset            : {final_images}")
    print(f"  Total labeled objects (boxes)      : {s['boxes_kept']}")
    print(f"  Output: {yolo_output_dir}")
    print("#" * 64)


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    config = load_and_process_config(args.config, PROJECT_ROOT)
    apply_cli_overrides(args, config)
    validate_component_selection(args, config)

    # SimulationApp must be created before any omni.* import inside the tasks.
    simulation_app = SimulationApp(asdict(config.SIMULATION_CONFIG))

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using provided seed: {args.seed}")

    try:
        # ---- 1. GENERATE -----------------------------------------------------
        print("--- Starting Isaac Sim Generation Task ---")
        isaac_output_dir = run_generation_task(args=args, config=config, simulation_app=simulation_app)
        print(f"--- Isaac Sim Generation Task Finished. Output at: {isaac_output_dir} ---")

        # ---- 2. CONVERT + per-object quality filters -------------------------
        yolo_output_dir = None
        conv_stats = None
        overlap_deleted = None
        iou_threshold = args.iou_threshold if args.iou_threshold is not None else config.YOLO_FILTERING.IOU_THRESHOLD

        if args.convert_to_yolo:
            if isaac_output_dir:
                yolo_output_dir, conv_stats = run_yolo_conversion(
                    source_dir=isaac_output_dir,
                    config=config,
                    light_mode=args.light_mode,
                    min_visibility=args.min_visibility,
                    drop_truncated=args.drop_truncated,
                    truncation_policy=args.truncation_policy,
                    occlusion_policy=args.occlusion_policy,
                )
            else:
                print("Skipping YOLO conversion because the generation task failed.")

        # ---- 3. Whole-frame overlap filter ------------------------------------
        if args.filter_overlaps:
            if yolo_output_dir:
                overlap_deleted = filter_overlapping_boxes(
                    yolo_dataset_path=yolo_output_dir,
                    iou_threshold=iou_threshold,
                )
            else:
                print("Skipping overlap filtering because YOLO conversion was not run or failed.")

        # ---- 4. REPORT --------------------------------------------------------
        if conv_stats is not None:
            print_final_summary(args, conv_stats, overlap_deleted, iou_threshold, yolo_output_dir)

    except Exception:
        import traceback
        print("\nFATAL ERROR: An exception occurred during the pipeline.", file=sys.stderr)
        traceback.print_exc()
    finally:
        print("All tasks complete. Closing SimulationApp...")
        simulation_app.close()
        print("SimulationApp closed successfully.")
