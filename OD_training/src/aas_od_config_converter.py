import json
import yaml
import sys
from aas_core3 import jsonization
from aas_core3 import types as aas_types
from pathlib import Path


# --- (Keep StyledYamlDumper and AASConfigConverter classes exactly as they are) ---
# ... (StyledYamlDumper class code) ...
# ... (AASConfigConverter class code) ...

class StyledYamlDumper(yaml.SafeDumper):
    pass


def represent_quoted_str(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')


def represent_capital_bool(dumper, data):
    value = 'True' if data else 'False'
    return dumper.represent_scalar('tag:yaml.org,2002:bool', value)


StyledYamlDumper.add_representer(str, represent_quoted_str)
StyledYamlDumper.add_representer(bool, represent_capital_bool)


class AASConfigConverter:
    # ... (Your entire AASConfigConverter class implementation here) ...
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

    def _parse_element(self, element: aas_types.SubmodelElement) -> any:
        if isinstance(element, aas_types.SubmodelElementCollection):
            return {child.id_short: self._parse_element(child) for child in element.value} if element.value else {}
        if isinstance(element, aas_types.SubmodelElementList):
            if element.id_short == "LIGHT_POSITIONS":
                return [self._convert_value(json.loads(child.value)) for child in
                        element.value] if element.value else []
            elif element.id_short == "COMPONENT_USD_MAP":
                usd_map = {}
                if element.value:
                    for child in element.value:
                        if hasattr(child, 'id_short') and hasattr(child, 'value_id') and child.value_id and hasattr(
                                child.value_id, 'keys') and child.value_id.keys:
                            usd_map[child.id_short] = child.value_id.keys[0].value
                return usd_map
            return {child.id_short: self._parse_element(child) for child in element.value} if element.value else {}
        if isinstance(element, aas_types.Property):
            if ',' in str(element.value):
                try:
                    return [self._convert_value(v.strip()) for v in element.value.split(',')]
                except Exception:
                    return self._convert_value(element.value)
            return self._convert_value(element.value)
        return getattr(element, 'value', None)

    def _post_process_config(self, config: dict) -> dict:
        """
        Restructures, reorders, and ENRICHES the generated dictionary.
        """
        # --- FIX: Ensure config_copy is defined at the beginning ---
        config_copy = config.copy()
        ordered_config = {}
        # ... (your existing for loop) ...

        # --- ADD THIS NEW LOGIC ---
        # After creating the ordered_config, intelligently add the missing key.
        # This assumes the data.yaml is always in the base_data_dir.
        if "paths" in ordered_config and "base_data_dir" in ordered_config["paths"]:
            base_dir = Path(ordered_config["paths"]["base_data_dir"])
            dataset_yaml_path = base_dir / "data.yaml"

            # Add the dynamically created key to the dictionary
            ordered_config["paths"]["dataset_yaml"] = str(dataset_yaml_path)

            print(f"--- Enriched config: Added 'dataset_yaml' path: {dataset_yaml_path} ---")

        # Now, add any remaining keys from the original config that weren't in desired_order
        ordered_config.update(config_copy)

        return ordered_config

    def get_config_dict(self, aas_id_short: str, sm_id_short: str) -> dict:
        target_aas = next((aas for aas in self.environment.asset_administration_shells if aas.id_short == aas_id_short),
                          None)
        if not target_aas: raise ValueError(f"AAS with id_short '{aas_id_short}' not found.")
        submodel_map = {sm.id: sm for sm in self.environment.submodels}
        target_submodel = None
        for sm_ref in target_aas.submodels:
            submodel = submodel_map.get(sm_ref.keys[0].value)
            if submodel and submodel.id_short == sm_id_short:
                target_submodel = submodel
                break
        if not target_submodel: raise ValueError(f"Submodel '{sm_id_short}' not found in AAS '{aas_id_short}'.")
        object_detection_element = next(
            (elem for elem in target_submodel.submodel_elements if elem.id_short == "ObjectDetection"), None)
        if not object_detection_element: raise ValueError(
            "Submodel 'OD_Module' does not contain the root element 'ObjectDetection'.")
        config_dict = self._parse_element(object_detection_element)
        return self._post_process_config(config_dict)


# --- NEW REUSABLE FUNCTION ---
def generate_config(aas_file_path: str, output_yaml_path: str):
    """
    Main function to run the AAS to YAML conversion.
    This is designed to be called from other scripts.
    """
    output_path = Path(output_yaml_path)
    aas_path = Path(aas_file_path)

    try:
        if not aas_path.is_file():
            raise FileNotFoundError(f"The input AAS file was not found at: {aas_path.resolve()}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        converter = AASConfigConverter(aas_file_path=str(aas_path))
        final_config_dict = converter.get_config_dict(
            aas_id_short="CA_Modul",
            sm_id_short="OD_Module"
        )

        with open(output_path, "w") as f:
            yaml.dump(final_config_dict, f, Dumper=StyledYamlDumper, default_flow_style=False, sort_keys=False,
                      indent=2)

        print(f"✅ Configuration successfully generated from '{aas_path.name}' and saved to '{output_path}'")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"❌ FATAL ERROR during config generation: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred during config generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# --- Simplified main execution block for standalone use ---
if __name__ == "__main__":
    # This part only runs when you execute this script directly
    # It's useful for testing the converter on its own.
    print("Running AAS converter as a standalone script...")
    AAS_FILE = "../AAS/CA_Modul-OD_pipeline_v0.5.json"
    OUTPUT_YAML_FILE = "cl_src/config/config_final.yaml"

    generate_config(aas_file_path=AAS_FILE, output_yaml_path=OUTPUT_YAML_FILE)

    print("\n--- Generated YAML Content ---")
    print(Path(OUTPUT_YAML_FILE).read_text())