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

import json
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


# ---------------------------------------------------------------------------
#  AAS -> SDG config (opt-in): read the SDG parameters from an AAS file so the
#  passport, not config.yaml, drives a run. config.yaml stays the fallback for
#  any section the AAS does not carry, so nothing breaks if a field is absent.
#
#  Only the cleanly-typed, faithfully-mirrored sections are sourced from the AAS:
#  generation_params (scene/spawning/camera), YOLO_CONVERSION, YOLO_FILTERING,
#  SIMULATION_CONFIG. The class registry (COMPONENT_USD_MAP, stored in valueId)
#  and the light tables remain config.yaml-authoritative.
# ---------------------------------------------------------------------------

# Full set of SDG config sections that a faithful AAS (v3.0+) carries. v2.0 only
# mirrors a subset cleanly; the converter sources whatever is present and lets
# config.yaml fill the rest, so this is safe against either template version.
_AAS_SDG_SECTIONS = ("paths", "generation_params", "COMPONENT_USD_MAP", "YOLO_CONVERSION",
                     "YOLO_FILTERING", "LIGHT_INTENSITY_MAP", "LIGHT_TEMPERATURE_MEAN",
                     "LIGHT_TEMPERATURE_STD", "LIGHT_POSITIONS", "SIMULATION_CONFIG",
                     "RTX_CARB_SETTINGS")


def _aas_find(node, id_short):
    if isinstance(node, dict):
        if node.get("idShort") == id_short:
            return node
        for v in node.values():
            r = _aas_find(v, id_short)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _aas_find(v, id_short)
            if r is not None:
                return r
    return None


def _aas_coerce(prop):
    """AAS Property -> typed Python value, honouring valueType.
    Handles two extra cases so the same reader works for v2.0 and v3.0:
      - value blank but valueId present (COMPONENT_USD_MAP stores USD in valueId),
      - value is a JSON array/object string (list-valued leaves like
        LIGHT_INTENSITY_MAP entries / LIGHT_POSITIONS)."""
    vt = (prop.get("valueType") or "").lower()
    val = prop.get("value")
    if (val is None or val == "") and prop.get("valueId"):
        try:
            return prop["valueId"]["keys"][0]["value"]
        except (KeyError, IndexError, TypeError):
            pass
    if val is None:
        return None
    if isinstance(val, str) and val[:1] in "[{":
        try:
            return json.loads(val)
        except ValueError:
            pass
    if vt == "xs:boolean":
        return str(val).strip().lower() == "true"
    if vt in ("xs:int", "xs:integer", "xs:long"):
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return val
    if vt in ("xs:float", "xs:double", "xs:decimal"):
        try:
            return float(val)
        except (TypeError, ValueError):
            return val
    return val


def _build_aas_node(id_short, value):
    """config.yaml value -> faithful AAS SubmodelElement node (the inverse of the
    reader above): dict -> Collection, list -> JSON-in-value Property, scalar ->
    typed Property. Round-trips through aas_sdg_overrides()."""
    if isinstance(value, dict):
        return {"idShort": id_short, "modelType": "SubmodelElementCollection",
                "value": [_build_aas_node(k, v) for k, v in value.items()]}
    if isinstance(value, list):
        return {"idShort": id_short, "valueType": "xs:string",
                "value": json.dumps(value), "modelType": "Property"}
    if isinstance(value, bool):
        return {"idShort": id_short, "valueType": "xs:boolean",
                "value": "true" if value else "false", "modelType": "Property"}
    if isinstance(value, int):
        return {"idShort": id_short, "valueType": "xs:integer",
                "value": str(value), "modelType": "Property"}
    if isinstance(value, float):
        return {"idShort": id_short, "valueType": "xs:float",
                "value": repr(value), "modelType": "Property"}
    return {"idShort": id_short, "valueType": "xs:string",
            "value": "" if value is None else str(value), "modelType": "Property"}


def sdg_config_to_aas_nodes(raw_config) -> list:
    """Build faithful AAS nodes for every SDG section present in a config.yaml
    dict — used to generate the v3.0 template's 'SmartSDG' collection so the AAS
    is a 1:1 source for SDG (proven by the round-trip self-test)."""
    return [_build_aas_node(s, raw_config[s]) for s in _AAS_SDG_SECTIONS
            if raw_config.get(s) is not None]


def _aas_parse(node):
    """Parse an AAS SubmodelElement (dict form) into a plain Python value:
    Collection/List -> dict keyed by child idShort; Property -> typed scalar."""
    mt = node.get("modelType")
    if mt in ("SubmodelElementCollection", "SubmodelElementList"):
        out = {}
        for child in node.get("value") or []:
            if isinstance(child, dict) and child.get("idShort"):
                out[child["idShort"]] = _aas_parse(child)
        return out
    if mt == "Property":
        return _aas_coerce(node)
    return node.get("value")


def aas_sdg_overrides(aas_path: str) -> dict:
    """Extract the faithfully-mirrored SDG sections from an AAS file as a
    config.yaml-shaped dict. Returns {} on any problem (caller falls back to
    config.yaml)."""
    try:
        data = json.loads(Path(aas_path).read_text(encoding="utf-8-sig"))
    except Exception as e:
        print(f"[config] could not read AAS '{aas_path}': {e}; using config.yaml")
        return {}
    overrides = {}
    for section in _AAS_SDG_SECTIONS:
        node = _aas_find(data, section)
        if node is not None:
            parsed = _aas_parse(node)
            if parsed is not None:   # accept scalars, lists, and empty dicts too
                overrides[section] = parsed
    return overrides


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` into `base` (over wins); returns base."""
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        elif v is not None:
            base[k] = v
    return base


def load_and_process_config(config_path: str, project_root: Path, aas_path: str = None) -> AppConfig:
    """Load the YAML configuration file into a structured AppConfig object.

    If `aas_path` is given, the SDG sections found in that AAS file override the
    matching config.yaml values (config.yaml fills any gaps) — so the passport
    drives the run while staying robust if the AAS lacks a field."""
    print(f"Loading configuration from: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    if aas_path:
        overrides = aas_sdg_overrides(aas_path)
        if overrides:
            print(f"  > Applying SDG config from AAS: {', '.join(overrides)} "
                  f"(config.yaml fills the rest) [{aas_path}]")
            _deep_merge(raw_config, overrides)

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
