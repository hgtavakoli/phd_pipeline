import json
from pathlib import Path
import yaml
import sys
import argparse  # <-- 1. Import argparse
from aas_core3 import jsonization
from aas_core3 import types as aas_types

# (The StyledYamlDumper and AASConfigConverter classes remain exactly the same...)
# ... PASTE YOUR EXISTING StyledYamlDumper and AASConfigConverter classes here ...
class StyledYamlDumper(yaml.SafeDumper):
    """
    A custom YAML dumper to enforce specific formatting rules:
    1. All strings are wrapped in double quotes.
    2. Booleans are represented as 'True' and 'False'.
    """
    pass


def represent_quoted_str(dumper, data):
    """Represents strings as double-quoted scalars."""
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')


def represent_capital_bool(dumper, data):
    """Represents booleans as 'True' or 'False'."""
    value = 'True' if data else 'False'
    return dumper.represent_scalar('tag:yaml.org,2002:bool', value)


# Add the custom representers to our dumper class
StyledYamlDumper.add_representer(str, represent_quoted_str)
StyledYamlDumper.add_representer(bool, represent_capital_bool)


# ==============================================================================
# --- Main AAS Converter Class ---
# ==============================================================================
class AASConfigConverter:
    """
    Reads an AAS file and converts a specific submodel into a perfectly
    formatted YAML configuration file.
    """

    def __init__(self, aas_file_path: str):
        self.aas_file_path = aas_file_path
        self.environment = self._load_environment()

    def _load_environment(self) -> aas_types.Environment:
        try:
            with open(self.aas_file_path, "r", encoding="utf-8") as f:
                jsonable = json.load(f)
            return jsonization.environment_from_jsonable(jsonable)
        except FileNotFoundError:
            raise FileNotFoundError(f"The AAS file was not found at '{self.aas_file_path}'.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while parsing the AAS file: {e}")

    def _convert_value(self, value: str):
        if isinstance(value, str):
            val_lower = value.lower()
            if val_lower == 'true': return True
            if val_lower == 'false': return False
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
        return value

    # --- MODIFIED SECTION 1: _parse_element ---
    def _parse_element(self, element: aas_types.SubmodelElement) -> any:
        if isinstance(element, aas_types.SubmodelElementCollection):
            return {child.id_short: self._parse_element(child) for child in element.value} if element.value else {}

        if isinstance(element, aas_types.SubmodelElementList):
            # Special case for LIGHT_POSITIONS
            if element.id_short == "LIGHT_POSITIONS":
                return [self._convert_value(json.loads(child.value)) for child in
                        element.value] if element.value else []

            # Special case for COMPONENT_USD_MAP to parse from valueId reference
            elif element.id_short == "COMPONENT_USD_MAP":
                usd_map = {}
                if element.value:
                    for child in element.value:
                        # Check for the expected structure to avoid errors
                        if (
                                hasattr(child, 'id_short') and
                                hasattr(child, 'value_id') and child.value_id and
                                hasattr(child.value_id, 'keys') and child.value_id.keys
                        ):
                            usd_map[child.id_short] = child.value_id.keys[0].value
                return usd_map

            # Default handling for other lists
            merged_dict = {child.id_short: self._parse_element(child) for child in
                           element.value} if element.value else {}
            return merged_dict

        if isinstance(element, aas_types.Property):
            if ',' in str(element.value):
                try:
                    return [self._convert_value(v.strip()) for v in element.value.split(',')]
                except Exception:
                    return self._convert_value(element.value)
            return self._convert_value(element.value)

        return getattr(element, 'value', None)

    # --- MODIFIED SECTION 2: _post_process_config ---
    def _post_process_config(self, config: dict) -> dict:
        """
        Restructures and reorders the generated dictionary to match the final
        target format and key order.
        """
        # Define the exact order of keys for the final YAML file.
        desired_order = [
            "paths",
            "generation_params",
            "Spawning",
            "camera",
            "COmponent",
            "COMPONENT_USD_MAP",  # Added the new key to the order
            "YOLO_conversion",
            "YOLO_filtering",
            "LIGHT_INTENSITY_MAP",
            "LIGHT_TEMPERATURE_MEAN",
            "LIGHT_TEMPERATURE_STD",
            "LIGHT_POSITIONS",
            "SIMULATION_CONFIG"
        ]

        config_copy = config.copy()
        ordered_config = {}

        light_temp_data = config_copy.pop("LIGHT_TEMPERATURE", None)

        for key in desired_order:
            if key == "LIGHT_TEMPERATURE_MEAN":
                if light_temp_data and "Mean" in light_temp_data:
                    ordered_config[key] = light_temp_data["Mean"]
            elif key == "LIGHT_TEMPERATURE_STD":
                if light_temp_data and "Std" in light_temp_data:
                    ordered_config[key] = light_temp_data["Std"]
            elif key in config_copy:
                ordered_config[key] = config_copy.pop(key)

        ordered_config.update(config_copy)
        return ordered_config

    def get_config_dict(self, aas_id_short: str, sm_id_short: str) -> dict:
        target_aas = next((aas for aas in self.environment.asset_administration_shells if aas.id_short == aas_id_short),
                          None)
        if not target_aas:
            raise ValueError(f"AAS with id_short '{aas_id_short}' not found.")

        submodel_map = {sm.id: sm for sm in self.environment.submodels}
        target_submodel = None
        for sm_ref in target_aas.submodels:
            submodel = submodel_map.get(sm_ref.keys[0].value)
            if submodel and submodel.id_short == sm_id_short:
                target_submodel = submodel
                break

        if not target_submodel:
            raise ValueError(f"Submodel '{sm_id_short}' not found in AAS '{aas_id_short}'.")

        smart_sdg_element = next((elem for elem in target_submodel.submodel_elements if elem.id_short == "SmartSDG"),
                                 None)
        if not smart_sdg_element:
            raise ValueError("Submodel does not contain the root element 'SmartSDG'.")

        config_dict = self._parse_element(smart_sdg_element)
        final_config = self._post_process_config(config_dict)
        return final_config

# ==============================================================================
# --- Main execution block ---
# ==============================================================================
if __name__ == "__main__":
    # --- 2. Set up Argument Parser ---
    parser = argparse.ArgumentParser(description="Convert an AAS JSON file to a YAML config for SmartSDG.")
    parser.add_argument("aas_input_file", help="Path to the input AAS JSON file.")
    parser.add_argument(
        "--output",
        "-o",
        default="src/config/config_final.yaml", # Default output path is now relative to the project root (where the script is run from)
        help="Path to save the generated YAML config file."
    )
    args = parser.parse_args()

    # --- 3. Use arguments instead of hardcoded paths ---
    aas_path = Path(args.aas_input_file)
    output_path = Path(args.output)


    try:
        # The rest of the logic remains the same
        if not aas_path.is_file():
            raise FileNotFoundError(f"The input AAS file was not found at the specified path: {aas_path.resolve()}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Reading AAS file from: '{aas_path}'")
        converter = AASConfigConverter(aas_file_path=str(aas_path))
        final_config_dict = converter.get_config_dict(
            aas_id_short="CA_Modul",
            sm_id_short="OD_Module"
        )

        with open(output_path, "w") as f:
            yaml.dump(
                final_config_dict,
                f,
                Dumper=StyledYamlDumper,
                default_flow_style=False,
                sort_keys=False,
                indent=2
            )

        print(f"✅ Configuration successfully generated and saved to '{output_path}'")
        # Print the absolute path to standard output so the batch script can capture it.
    # Correctly structured exception handling starts here
    except FileNotFoundError as e:
        print(f"[FATAL ERROR] A required file was not found.")
        print(f"   Details: {e}")
        sys.exit(1)

    except (ValueError, RuntimeError) as e:
        print(f"[FATAL ERROR] An error occurred during AAS processing.")
        print(f"   This might be due to incorrect content in the AAS file (e.g., missing IDs).")
        print(f"   Details: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"[FATAL ERROR] An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)