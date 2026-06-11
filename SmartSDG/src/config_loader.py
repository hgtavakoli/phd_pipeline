"""
Dataclass schema for config.yaml.

Pure data definitions, no logic — the loading/validation lives in
config_process.load_and_process_config. Fields with a default of None are
populated by the loader (resolved absolute paths, derived class names).
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any

# -- Nested dataclasses mirroring the YAML structure --

@dataclass
class PathsConfig:
    base: str
    output_raw: str
    env_usd_rel: str
    flat_env_usd_rel: str
    components_rel: str
    # These fields will be populated by the loader function
    env_usd_abs: Path = None
    flat_env_usd_abs: Path = None
    components_abs: Path = None
    output_raw_abs: Path = None

@dataclass
class SceneConfig:
    center_x: float
    center_y: float
    truck_z_position: float
    truck_scale: float

@dataclass
class SpawningConfig:
    multi_spawn_radius: float
    multi_spawn_z_variation: float

@dataclass
class CameraConfig:
    img_width: int
    img_height: int
    focal_length: float
    focus_distance: float
    f_stop: float
    orbit_radius: float
    orbit_height: float
    orbit_height_isolated: float

@dataclass
class GenerationParamsConfig:
    scene: SceneConfig
    spawning: SpawningConfig
    camera: CameraConfig

@dataclass
class YOLOConversionConfig:
    OUTPUT_DIR_BASE: str
    # This field will be populated by the loader function
    CLASS_NAMES: List[str] = field(default_factory=list)
    output_dir_abs: Path = None

@dataclass
class YOLOFilteringConfig:
    enabled: bool
    IOU_THRESHOLD: float

@dataclass
class SimulationConfig:
    headless: bool
    renderer: str
    samples_per_pixel_per_frame: int
    max_bounces: int
    denoiser: bool
    use_denoiser: bool
    # Optional ray-tracing hyperparameters (SimulationApp launcher keys)
    max_specular_transmission_bounces: int = 6
    max_volume_bounces: int = 4
    subdiv_refinement_level: int = 0
    anti_aliasing: int = 3

# -- Main Configuration Dataclass --

@dataclass
class AppConfig:
    paths: PathsConfig
    generation_params: GenerationParamsConfig
    COMPONENT_USD_MAP: Dict[str, str]
    YOLO_CONVERSION: YOLOConversionConfig
    YOLO_FILTERING: YOLOFilteringConfig
    LIGHT_INTENSITY_MAP: Dict[str, List[int]]
    LIGHT_TEMPERATURE_MEAN: int
    LIGHT_TEMPERATURE_STD: int
    LIGHT_POSITIONS: List[List[float]]
    SIMULATION_CONFIG: SimulationConfig
    # Free-form carb settings applied at generation time, e.g.
    # "/rtx/pathtracing/maxSamplesPerLaunch": 1048576
    RTX_CARB_SETTINGS: Dict[str, Any] = field(default_factory=dict)
    # These fields will be populated by the loader function
    SINGLE_SPAWN_BOX_MIN: List[float] = field(default_factory=list)
    SINGLE_SPAWN_BOX_MAX: List[float] = field(default_factory=list)
    MULTI_SPAWN_AREA_MIN: List[float] = field(default_factory=list)
    MULTI_SPAWN_AREA_MAX: List[float] = field(default_factory=list)