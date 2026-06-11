# ==============================================================================
#
# OMNIVERSE ISAAC SIM - UNIFIED SYNTHETIC DATA GENERATION SCRIPT (V5 - RANDOM ORBIT)
#
# ==============================================================================
#
# DESCRIPTION:
# This version updates the single-object mode to use a random orbital camera
# for `camera2`, providing varied viewpoints similar to multi-object mode,
# while keeping `camera1` in its fixed top-down position for consistency.
#
# ==============================================================================

import argparse
import math
import os
import random
import sys
from typing import Dict, List, Tuple

import carb.settings
import numpy as np

try:
    from omni.isaac.kit import SimulationApp
except ImportError:
    print("Could not import SimulationApp from omni.isaac.kit, trying from isaacsim...", file=sys.stderr)
    try:
        from isaacsim.simulation_app import SimulationApp
    except ImportError:
        print("CRITICAL ERROR: Could not import SimulationApp. Please ensure Isaac Sim is installed correctly.", file=sys.stderr)
        sys.exit(1)


class Config:
    """A centralized configuration class to hold all script parameters."""
    BASE_PATH = "/home/incorap/hooman/SmartSDG/assets"
    ENV_USD_PATH = os.path.join(BASE_PATH, "env/SF_model/SDG_SF_Fixed_model_v5.usd")
    FLAT_ENV_USD_PATH = os.path.join(BASE_PATH, "env/SF_model/flattened_saved.usd")
    TRUCK_COMPONENTS_PATH = os.path.join(BASE_PATH, "Incorap_lkw")
    OUTPUT_DIR_BASE = "/home/incorap/hooman/composer-workspace/unified_render_output/"
    COMPONENT_USD_MAP: Dict[str, str] = {
        'Ylid': "YellowkLid.usd", 'YTrailer': "YellowTrailer.usd", 'Cabin': "Cabin.usd",
        'GlueGun': "GlueGun_new.usd", 'BLid': "Bluelid.usd", 'BTrailer': "BlueTrailer.usd",
        'FChassis': "Fchassis.usd", 'Rchassis': "Rchassis.usd"
    }
    LIGHT_INTENSITY_MAP: Dict[str, Tuple[int, int]] = {"S": (5000, 500), "M": (20000, 5000), "L": (80000, 20000)}
    LIGHT_TEMPERATURE_MEAN, LIGHT_TEMPERATURE_STD = 8000, 500
    LIGHT_POSITIONS: List[Tuple[float, float, float]] = [
        (-1, 18.62, 3.95), (-1, 20.62, 3.95), (-1, 22.62, 3.95), (-1, 24.62, 3.95),
        (-7, 18.62, 3.95), (-7, 20.62, 3.95), (-7, 22.62, 3.95), (-7, 24.62, 3.95),
        (-5, 18.62, 3.95), (-5, 20.62, 3.95), (-5, 22.62, 3.95), (-5, 24.62, 3.95),
        (-3, 18.62, 3.95), (-3, 20.62, 3.95), (-3, 22.62, 3.95), (-3, 24.62, 3.95)
    ]
    SCENE_CENTER_X, SCENE_CENTER_Y, TRUCK_Z_POSITION, TRUCK_SCALE = -4.5, 23.0, 0.95, 0.01
    MULTI_SPAWN_RADIUS, MULTI_SPAWN_Z_VARIATION = 0.3, 0.3
    SINGLE_SPAWN_BOX_MIN = [SCENE_CENTER_X, SCENE_CENTER_Y, TRUCK_Z_POSITION]
    SINGLE_SPAWN_BOX_MAX = [SCENE_CENTER_X, SCENE_CENTER_Y, TRUCK_Z_POSITION + 0.04]
    MULTI_SPAWN_AREA_MIN = [SCENE_CENTER_X - MULTI_SPAWN_RADIUS, SCENE_CENTER_Y - MULTI_SPAWN_RADIUS, TRUCK_Z_POSITION]
    MULTI_SPAWN_AREA_MAX = [SCENE_CENTER_X + MULTI_SPAWN_RADIUS, SCENE_CENTER_Y + MULTI_SPAWN_RADIUS, TRUCK_Z_POSITION + MULTI_SPAWN_Z_VARIATION]
    IMG_WIDTH, IMG_HEIGHT = 128, 80
    CAMERA_FOCAL_LENGTH, CAMERA_FOCUS_DISTANCE, CAMERA_F_STOP = 48.0, 400.0, 0.0
    # === V5 CHANGE: Parameters for orbital cameras in BOTH modes ===
    ORBIT_RADIUS = 1.5
    ORBIT_HEIGHT = 1.0
    SIMULATION_CONFIG: Dict = {
        "headless": True, "renderer": "PathTracing", "width": IMG_WIDTH, "height": IMG_HEIGHT,
        "samples_per_pixel_per_frame": 64, "max_bounces": 32, "denoiser": True, "use_denoiser": True,
    }

# === V5 CHANGE: Updated function to be more flexible ===
def generate_orbital_camera_position(orbit_center: Tuple[float, float, float], radius: float, height: float) -> Tuple[float, float, float]:
    """Calculates a random camera position on a circular path around a given center."""
    angle = random.uniform(0, 2 * math.pi)
    cam_x = orbit_center[0] + radius * math.cos(angle)
    cam_y = orbit_center[1] + radius * math.sin(angle)
    cam_z = height
    return (cam_x, cam_y, cam_z)

def run_generation_task(
        component_class: str, lighting_mode: str, num_samples: int, load_env: str,
        multi_object_flag: bool, config: Config, simulation_app: SimulationApp
) -> None:
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Sdf

    stage = omni.usd.get_context().get_stage()
    default_light_path = Sdf.Path("/Environment/defaultLight")
    if stage.GetPrimAtPath(default_light_path).IsValid():
        stage.RemovePrim(default_light_path)
        print("Removed default environment light.")

    env_suffix_map = {'true': 'withenv', 'plane': 'plane', 'false': 'withoutenv'}
    env_suffix = env_suffix_map[load_env]
    mode_prefix = 'MULTI' if multi_object_flag else 'SINGLE'
    component_name = 'ALL_CLASSES' if multi_object_flag else component_class
    folder_name = f"{mode_prefix}_{component_name}_{lighting_mode}_{env_suffix}"
    output_dir = os.path.join(config.OUTPUT_DIR_BASE, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting Generation ---")
    print(f"  > Mode: {'Multi-Object' if multi_object_flag else 'Single-Object'}")
    print(f"  > Component: {component_name}")
    print(f"  > Samples: {num_samples}")
    print(f"  > Output Directory: {os.path.abspath(output_dir)}")
    carb.settings.get_settings().set("/exts/omni.replicator.core/maxAssetLoadingTime", 1200)

    with rep.new_layer():
        if load_env == 'true':
            env = rep.create.from_usd(config.ENV_USD_PATH)
            with env: rep.modify.pose(position=(0, 0, 0), scale=(1, 1, 1))
        elif load_env == 'plane':
            env = rep.create.from_usd(config.FLAT_ENV_USD_PATH)
            with env: rep.modify.pose(position=(0, 0, 0), scale=(1, 1, 1))
        else:
            print("No environment will be loaded.")

        def randomize_objects():
            if multi_object_flag:
                with rep.create.xform() as group:
                    for name, model in config.COMPONENT_USD_MAP.items():
                        with rep.create.from_usd(os.path.join(config.TRUCK_COMPONENTS_PATH, model), semantics=[('class', name)]):
                            rep.modify.pose(position=rep.distribution.uniform(config.MULTI_SPAWN_AREA_MIN, config.MULTI_SPAWN_AREA_MAX),
                                            rotation=rep.distribution.uniform((-45, -45, -45), (45, 45, 45)), scale=config.TRUCK_SCALE)
                return group.node
            else:
                usd_path = os.path.join(config.TRUCK_COMPONENTS_PATH, config.COMPONENT_USD_MAP[component_class])
                with rep.create.from_usd(usd_path, semantics=[('class', component_class)]) as truck:
                    rep.modify.pose(position=rep.distribution.uniform(config.SINGLE_SPAWN_BOX_MIN, config.SINGLE_SPAWN_BOX_MAX),
                                    rotation=rep.distribution.uniform((-180, -180, -180), (180, 180, 180)), scale=config.TRUCK_SCALE)
                return truck.node
        rep.randomizer.register(randomize_objects)

        def randomize_lights():
            mean, std = config.LIGHT_INTENSITY_MAP[lighting_mode]
            lights = rep.create.group([])
            with lights:
                for pos in config.LIGHT_POSITIONS:
                    rep.create.light(light_type="Rect", temperature=rep.distribution.normal(config.LIGHT_TEMPERATURE_MEAN, config.LIGHT_TEMPERATURE_STD),
                                     intensity=rep.distribution.normal(mean, std), position=pos)
            return lights.node
        rep.randomizer.register(randomize_lights)

        camera1 = rep.create.camera(focal_length=config.CAMERA_FOCAL_LENGTH, focus_distance=config.CAMERA_FOCUS_DISTANCE, f_stop=config.CAMERA_F_STOP)
        camera2 = rep.create.camera(focal_length=config.CAMERA_FOCAL_LENGTH, focus_distance=config.CAMERA_FOCUS_DISTANCE, f_stop=config.CAMERA_F_STOP)

        # === V5 FIX: New camera logic block ===
        if multi_object_flag:
            print("Using DYNAMIC camera positions for multi-object mode.")
            multi_orbit_center = (config.SCENE_CENTER_X, config.SCENE_CENTER_Y, config.TRUCK_Z_POSITION + (config.MULTI_SPAWN_Z_VARIATION / 2))
            cam1_pos_sequence = [(config.SCENE_CENTER_X, config.SCENE_CENTER_Y, random.uniform(2.5, 3.0)) for _ in range(num_samples)]
            cam2_pos_sequence = [generate_orbital_camera_position(multi_orbit_center, config.ORBIT_RADIUS, config.ORBIT_HEIGHT) for _ in range(num_samples)]
            look_at_target_static = multi_orbit_center
        else:
            print("Using FIXED top-down camera and RANDOM orbital camera for single-object mode.")
            # Camera 1 remains fixed for consistency
            cam1_pos_sequence = [(-4.0, 23.0, 1.95)] * num_samples
            
            # Camera 2 now generates random positions in an orbit around the single object's center
            single_orbit_center = (config.SCENE_CENTER_X, config.SCENE_CENTER_Y, config.TRUCK_Z_POSITION)
            cam2_pos_sequence = [generate_orbital_camera_position(single_orbit_center, config.ORBIT_RADIUS, config.ORBIT_HEIGHT) for _ in range(num_samples)]
            look_at_target_static = None

        with rep.trigger.on_frame(max_execs=num_samples):
            rep.randomizer.randomize_objects()
            rep.randomizer.randomize_lights()

            if multi_object_flag:
                look_at_target = look_at_target_static
            else:
                look_at_target = rep.get.prims(semantics=[('class', component_class)])

            with camera1:
                rep.modify.pose(position=rep.distribution.sequence(cam1_pos_sequence), look_at=look_at_target)
            with camera2:
                rep.modify.pose(position=rep.distribution.sequence(cam2_pos_sequence), look_at=look_at_target)

        render_product1 = rep.create.render_product(camera1, (config.IMG_WIDTH, config.IMG_HEIGHT))
        render_product2 = rep.create.render_product(camera2, (config.IMG_WIDTH, config.IMG_HEIGHT))
        basic_writer = rep.WriterRegistry.get("BasicWriter")
        basic_writer.initialize(output_dir=output_dir, rgb=True, bounding_box_2d_tight=True)
        basic_writer.attach([render_product1, render_product2])

    print("Starting orchestrator...")
    rep.orchestrator.run()
    rep.orchestrator.wait_until_complete()
    print("Orchestrator finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified SDG Script for Single and Multi-Object Generation.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--multi_object", action="store_true", help="If set, generate scenes with all components. Otherwise, generate single-component scenes.")
    parser.add_argument("--component_class", type=str, default='Cabin', choices=list(Config.COMPONENT_USD_MAP.keys()), help="Specify the component to render in single-object mode. Ignored if --multi_object is set.")
    parser.add_argument("--light_mode", type=str, required=True, choices=['S', 'M', 'L'], help="Lighting mode: S (Small), M (Medium), L (Large) intensity.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of images to generate.")
    parser.add_argument("--load_env", type=str, default='false', choices=['true', 'plane', 'false'], help="Specify environment: 'true' for full scene, 'plane' for a flat plane, 'false' for no environment.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for random number generators.")
    args = parser.parse_args()

    cfg = Config()
    simulation_app = SimulationApp(cfg.SIMULATION_CONFIG)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using provided seed: {args.seed}")
    try:
        run_generation_task(
            component_class=args.component_class, lighting_mode=args.light_mode, num_samples=args.num_samples,
            load_env=args.load_env, multi_object_flag=args.multi_object, config=cfg, simulation_app=simulation_app
        )
    except Exception as e:
        print(f"\nFATAL ERROR: An exception occurred during the simulation.", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        print("Closing SimulationApp...")
        simulation_app.close()
        print("SimulationApp closed successfully.")