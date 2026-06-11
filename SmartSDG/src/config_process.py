"""
Config loading: YAML file -> structured, validated AppConfig object.

`load_and_process_config` is the single entry point. Beyond parsing it also:
  * derives CLASS_NAMES from COMPONENT_USD_MAP order (this fixes the global
    YOLO class IDs — never reorder the map in config.yaml),
  * resolves all relative paths against the project root and verifies that the
    asset directories exist,
  * pre-computes the single/multi object spawn boxes,
  * syncs the simulation viewport size to the camera resolution.
"""

import os
import sys
from pathlib import Path

import yaml

# Allow `from config_loader import ...` when this folder is not on sys.path.
SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config_loader import (AppConfig, CameraConfig, GenerationParamsConfig, PathsConfig,
                           SceneConfig, SimulationConfig, SpawningConfig,
                           YOLOConversionConfig, YOLOFilteringConfig)


def load_and_process_config(config_path: str, project_root: Path) -> AppConfig:
    """Load the YAML configuration file into a structured AppConfig object."""
    print(f"Loading configuration from: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # --- Validate required sections ---
    if not raw_config.get('paths'):
        raise KeyError("Configuration Error: The 'paths' section is missing from your config.yaml file.")

    yolo_data = raw_config.get('YOLO_CONVERSION')
    if not yolo_data:
        raise KeyError("Configuration Error: 'YOLO_CONVERSION' section is missing from config.yaml.")

    # CLASS_NAMES (and therefore the YOLO class IDs 0..N) are defined by the
    # order of COMPONENT_USD_MAP — keep that order stable across all datasets.
    yolo_data['CLASS_NAMES'] = list(raw_config['COMPONENT_USD_MAP'].keys())

    # --- Instantiate the dataclass tree ---
    config = AppConfig(
        paths=PathsConfig(**raw_config['paths']),
        generation_params=GenerationParamsConfig(
            scene=SceneConfig(**raw_config['generation_params']['scene']),
            spawning=SpawningConfig(**raw_config['generation_params']['spawning']),
            camera=CameraConfig(**raw_config['generation_params']['camera'])
        ),
        COMPONENT_USD_MAP=raw_config['COMPONENT_USD_MAP'],
        YOLO_CONVERSION=YOLOConversionConfig(**raw_config['YOLO_CONVERSION']),
        YOLO_FILTERING=YOLOFilteringConfig(**raw_config['YOLO_FILTERING']),
        LIGHT_INTENSITY_MAP=raw_config['LIGHT_INTENSITY_MAP'],
        LIGHT_TEMPERATURE_MEAN=raw_config['LIGHT_TEMPERATURE_MEAN'],
        LIGHT_TEMPERATURE_STD=raw_config['LIGHT_TEMPERATURE_STD'],
        LIGHT_POSITIONS=raw_config['LIGHT_POSITIONS'],
        SIMULATION_CONFIG=SimulationConfig(**raw_config['SIMULATION_CONFIG']),
        RTX_CARB_SETTINGS=raw_config.get('RTX_CARB_SETTINGS') or {}
    )

    # --- Resolve paths against the project root and verify assets exist ---
    base_path_abs = project_root / config.paths.base
    if not base_path_abs.exists():
        raise FileNotFoundError(
            f"Asset directory not found at: {base_path_abs}\n"
            f"Please ensure the '{config.paths.base}' folder exists in your project root."
        )
    config.paths.env_usd_abs = base_path_abs / config.paths.env_usd_rel
    config.paths.flat_env_usd_abs = base_path_abs / config.paths.flat_env_usd_rel
    config.paths.components_abs = base_path_abs / config.paths.components_rel
    if not config.paths.components_abs.exists():
        raise FileNotFoundError(
            f"Components directory not found at: {config.paths.components_abs}\n"
            f"Please download the required 3D models and place them in this directory."
        )
    config.paths.output_raw_abs = project_root / config.paths.output_raw
    config.YOLO_CONVERSION.output_dir_abs = project_root / config.YOLO_CONVERSION.OUTPUT_DIR_BASE

    # --- Pre-compute spawn boxes (may be overridden later by --spawn_radius) ---
    scene = config.generation_params.scene
    spawn = config.generation_params.spawning
    config.SINGLE_SPAWN_BOX_MIN = [scene.center_x, scene.center_y, scene.truck_z_position]
    config.SINGLE_SPAWN_BOX_MAX = [scene.center_x, scene.center_y, scene.truck_z_position + 0.04]
    config.MULTI_SPAWN_AREA_MIN = [scene.center_x - spawn.multi_spawn_radius,
                                   scene.center_y - spawn.multi_spawn_radius,
                                   scene.truck_z_position]
    config.MULTI_SPAWN_AREA_MAX = [scene.center_x + spawn.multi_spawn_radius,
                                   scene.center_y + spawn.multi_spawn_radius,
                                   scene.truck_z_position + spawn.multi_spawn_z_variation]

    # --- Keep the simulation viewport in sync with the camera resolution ---
    camera = config.generation_params.camera
    config.SIMULATION_CONFIG.width = camera.img_width
    config.SIMULATION_CONFIG.height = camera.img_height

    print("Configuration loaded and processed successfully.")
    return config
