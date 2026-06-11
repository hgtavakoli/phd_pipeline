"""
Isaac Sim Replicator scene generation.

Builds the Replicator graph for one dataset run:
  * loads the environment (full 3D scene / flat plane / void),
  * registers per-frame randomizers for object placement (optionally
    collision-aware via --min_separation) and for lighting,
  * sets up the dual-camera rig (camera 1: fixed top-down or dynamic overhead;
    camera 2: random orbital position) and the render products,
  * writes RGB + bounding_box_2d_tight via BasicWriter and runs the orchestrator.

Must run inside Isaac Sim (imports omni.* / carb after SimulationApp exists).
"""

import os
import random
import sys
from pathlib import Path

import carb.settings

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config_loader import AppConfig
from utils import generate_orbital_camera_position


def run_generation_task(
    args: "argparse.Namespace",
    config: AppConfig,  # <-- Type hint updated to use the AppConfig class
    simulation_app
) -> str:
    """
    Sets up and executes the entire Replicator pipeline for data generation.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        config (AppConfig): The structured and processed configuration object.
        simulation_app: The running SimulationApp instance.
    """
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Sdf

    # --- Use processed config parameters for better readability ---
    paths = config.paths
    gen_params = config.generation_params
    cam_params = gen_params.camera
    scene_params = gen_params.scene
    spawn_params = gen_params.spawning

    # --- INITIAL SCENE & DIRECTORY SETUP ---
    stage = omni.usd.get_context().get_stage()
    default_light_path = Sdf.Path("/Environment/defaultLight")
    if stage.GetPrimAtPath(default_light_path).IsValid():
        stage.RemovePrim(default_light_path)
        print("Removed default environment light.")

    env_suffix_map = {'true': 'withenv', 'plane': 'plane', 'false': 'withoutenv'}
    env_suffix = env_suffix_map[args.load_env]
    mode_prefix = 'MULTI' if args.multi_object else 'SINGLE'
    components_subset = getattr(args, 'components_list', None)
    if args.multi_object:
        component_name = '-'.join(components_subset) if components_subset else 'ALL_CLASSES'
    else:
        component_name = args.component_class
    folder_name = f"{mode_prefix}_{component_name}_{args.light_mode}_{env_suffix}"
    
    # Correctly use the processed 'output_raw_abs' path
    output_dir = os.path.join(str(paths.output_raw_abs), folder_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting Generation ---")
    print(f"  > Mode: {'Multi-Object' if args.multi_object else 'Single-Object'}")
    print(f"  > Component: {component_name}")
    print(f"  > Lighting Mode: {args.light_mode}")
    print(f"  > Environment State: {'SF_environment Loaded.' if args.load_env =='true' else 'Plane is Loaded.' if args.load_env == 'plane' else 'empty environment.'}")
    print(f"  > Samples: {args.num_samples}")
    print(f"  > Output Directory: {os.path.abspath(output_dir)}")
    carb.settings.get_settings().set("/exts/omni.replicator.core/maxAssetLoadingTime", 1200)

    # Apply user-defined RTX/Replicator carb settings from config.yaml
    rtx_settings = getattr(config, 'RTX_CARB_SETTINGS', None) or {}
    for key, value in rtx_settings.items():
        carb.settings.get_settings().set(key, value)
        print(f"  > carb setting: {key} = {value}")

    # --- DEFINE THE REPLICATOR GRAPH ---
    with rep.new_layer():
        if args.load_env == 'true':
            env = rep.create.from_usd(str(paths.env_usd_abs))
            with env: rep.modify.pose(position=(0, 0, 0), scale=(1, 1, 1))
        elif args.load_env == 'plane':
            env = rep.create.from_usd(str(paths.flat_env_usd_abs))
            with env: rep.modify.pose(position=(0, 0, 0), scale=(1, 1, 1))

        def _sample_separated_positions(num_frames, num_objects, min_dist):
            """Per-frame positions inside the spawn box with a minimum pairwise
            XY distance, via rejection sampling. Returns one position sequence
            per object (each of length num_frames)."""
            import math
            lo, hi = config.MULTI_SPAWN_AREA_MIN, config.MULTI_SPAWN_AREA_MAX
            sequences = [[] for _ in range(num_objects)]
            for _ in range(num_frames):
                placed = []
                for obj_idx in range(num_objects):
                    best = None
                    for _attempt in range(300):
                        cand = (random.uniform(lo[0], hi[0]), random.uniform(lo[1], hi[1]), random.uniform(lo[2], hi[2]))
                        if all(math.hypot(cand[0] - p[0], cand[1] - p[1]) >= min_dist for p in placed):
                            best = cand
                            break
                    if best is None:
                        # spawn box too small for this many objects at min_dist -> place at least overlap-minimal
                        best = max(((random.uniform(lo[0], hi[0]), random.uniform(lo[1], hi[1]), random.uniform(lo[2], hi[2]))
                                    for _ in range(50)),
                                   key=lambda c: min((math.hypot(c[0] - p[0], c[1] - p[1]) for p in placed), default=0))
                        print(f"  > WARNING: could not satisfy min_separation={min_dist} for all objects; spawn area may be too small.")
                    placed.append(best)
                    sequences[obj_idx].append(best)
            return sequences

        min_separation = getattr(args, 'min_separation', None)

        def randomize_objects():
            if args.multi_object:
                active_map = config.COMPONENT_USD_MAP
                if components_subset:
                    active_map = {k: config.COMPONENT_USD_MAP[k] for k in components_subset}
                position_sequences = None
                if min_separation:
                    print(f"Using collision-aware placement: min separation {min_separation} m between object centers.")
                    position_sequences = _sample_separated_positions(args.num_samples, len(active_map), min_separation)
                with rep.create.xform() as group:
                    for obj_idx, (name, model) in enumerate(active_map.items()):
                        # Use the new attribute-style access for paths
                        usd_path = os.path.join(str(paths.components_abs), model)
                        if position_sequences:
                            position_dist = rep.distribution.sequence(position_sequences[obj_idx])
                        else:
                            position_dist = rep.distribution.uniform(config.MULTI_SPAWN_AREA_MIN, config.MULTI_SPAWN_AREA_MAX)
                        with rep.create.from_usd(usd_path, semantics=[('class', name)]):
                            rep.modify.pose(
                                position=position_dist,
                                rotation=rep.distribution.uniform((-45, -45, -45), (45, 45, 45)),
                                scale=scene_params.truck_scale
                            )
                return group.node
            else:
                usd_path = os.path.join(str(paths.components_abs), config.COMPONENT_USD_MAP[args.component_class])
                with rep.create.from_usd(usd_path, semantics=[('class', args.component_class)]) as truck:
                    rep.modify.pose(
                        position=rep.distribution.uniform(config.SINGLE_SPAWN_BOX_MIN, config.SINGLE_SPAWN_BOX_MAX),
                        rotation=rep.distribution.uniform((-180, -180, -180), (180, 180, 180)),
                        scale=scene_params.truck_scale
                    )
                return truck.node
        rep.randomizer.register(randomize_objects)

        def randomize_lights():
            mean, std = config.LIGHT_INTENSITY_MAP[args.light_mode]
            lights = rep.create.group([])
            with lights:
                for pos in config.LIGHT_POSITIONS:
                    rep.create.light(
                        light_type="Rect",
                        temperature=rep.distribution.normal(config.LIGHT_TEMPERATURE_MEAN, config.LIGHT_TEMPERATURE_STD),
                        intensity=rep.distribution.normal(mean, std),
                        position=pos
                    )
            return lights.node
        rep.randomizer.register(randomize_lights)

        camera1 = rep.create.camera(focal_length=cam_params.focal_length, focus_distance=cam_params.focus_distance, f_stop=cam_params.f_stop)
        camera2 = rep.create.camera(focal_length=cam_params.focal_length, focus_distance=cam_params.focus_distance, f_stop=cam_params.f_stop)

        if args.multi_object:
            print("Using DYNAMIC camera positions for multi-object mode.")
            multi_orbit_center = (scene_params.center_x, scene_params.center_y, scene_params.truck_z_position + (spawn_params.multi_spawn_z_variation / 2))
            cam1_pos_sequence = [(scene_params.center_x, scene_params.center_y, random.uniform(2.5, 3.0)) for _ in range(args.num_samples)]
            cam2_pos_sequence = [generate_orbital_camera_position(multi_orbit_center, cam_params.orbit_radius, cam_params.orbit_height) for _ in range(args.num_samples)]
            look_at_target_static = multi_orbit_center
        else:
            print("Using FIXED top-down camera and RANDOM orbital camera for single-object mode.")
            cam1_pos_sequence = [(-4.0, 23.0, 1.95)] * args.num_samples
            
            active_orbit_height = cam_params.orbit_height
            if args.load_env == 'plane':
                print("...Plane detected. Using slightly higher orbital camera to avoid seeing the ground.")
                active_orbit_height = cam_params.orbit_height_isolated

            single_orbit_center = (scene_params.center_x, scene_params.center_y, scene_params.truck_z_position)
            
            cam2_pos_sequence = [
                generate_orbital_camera_position(
                    single_orbit_center,
                    cam_params.orbit_radius,
                    active_orbit_height
                ) for _ in range(args.num_samples)
            ]
            look_at_target_static = None

        with rep.trigger.on_frame(max_execs=args.num_samples):
            rep.randomizer.randomize_objects()
            rep.randomizer.randomize_lights()
            look_at_target = look_at_target_static if args.multi_object else rep.get.prims(semantics=[('class', args.component_class)])
            with camera1:
                rep.modify.pose(position=rep.distribution.sequence(cam1_pos_sequence), look_at=look_at_target)
            with camera2:
                rep.modify.pose(position=rep.distribution.sequence(cam2_pos_sequence), look_at=look_at_target)

        render_product1 = rep.create.render_product(camera1, (cam_params.img_width, cam_params.img_height))
        render_product2 = rep.create.render_product(camera2, (cam_params.img_width, cam_params.img_height))
        
        basic_writer = rep.WriterRegistry.get("BasicWriter")
        basic_writer.initialize(output_dir=output_dir, rgb=True, bounding_box_2d_tight=True)
        basic_writer.attach([render_product1, render_product2])

    print("Starting orchestrator...")
    rep.orchestrator.run()
    rep.orchestrator.wait_until_complete()
    print("Orchestrator finished.")

    return output_dir