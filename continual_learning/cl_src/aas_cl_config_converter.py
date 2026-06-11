# cl_src/aas_cl_config_converter.py

import json
import yaml
import sys
from aas_core3 import jsonization
from aas_core3 import types as aas_types
from pathlib import Path


class StyledYamlDumper(yaml.SafeDumper):
    pass


class AASConfigConverter:
    """
    A robust converter to parse specific configurations from a complex AAS Environment JSON file.
    """

    def __init__(self, aas_file_path: str):
        self.aas_file_path = Path(aas_file_path)
        self.environment = self._load_environment()

    def _load_environment(self) -> aas_types.Environment:
        try:
            with self.aas_file_path.open("r", encoding="utf-8") as f:
                jsonable = json.load(f)
            # The SDK expects a full Environment structure. If the file is just one
            # part (like a single Submodel), we wrap it to conform.
            if "submodels" not in jsonable and "assetAdministrationShells" not in jsonable:
                print(
                    "--> Info: JSON appears to be a single object. Attempting to parse as Submodel and wrap in an Environment.")
                # Assume the root object is the CA_Modul Submodel
                ca_modul_submodel = jsonization.submodel_from_jsonable(jsonable)

                # Create a dummy AAS to reference this submodel, which is required by the SDK structure.
                dummy_aas = aas_types.AssetAdministrationShell(
                    id=f"urn:dummy:aas:{ca_modul_submodel.id_short}",
                    id_short=ca_modul_submodel.id_short,
                    submodels=[aas_types.Reference(
                        type=aas_types.ReferenceTypes.MODEL_REFERENCE,
                        keys=[ca_modul_submodel.keys[0]]
                    )]
                )
                # Rebuild the jsonable object into the expected Environment format
                jsonable = {
                    "assetAdministrationShells": [jsonization.to_jsonable(dummy_aas)],
                    "submodels": [jsonization.to_jsonable(ca_modul_submodel)]
                }
            return jsonization.environment_from_jsonable(jsonable)
        except FileNotFoundError:
            raise FileNotFoundError(f"The AAS file was not found at '{self.aas_file_path}'.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while parsing the AAS file: {e}")

    def _convert_value(self, value: any):
        """Converts string values to appropriate Python types."""
        if isinstance(value, str):
            val_lower = value.lower()
            if val_lower == 'true': return True
            if val_lower == 'false': return False
            try:
                # Keep comma-separated strings as-is for later splitting
                if ',' in value:
                    return value
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
        return value

    def _parse_element(self, element: aas_types.SubmodelElement) -> any:
        """Recursively parses AAS elements into a Python dictionary."""
        if isinstance(element, (aas_types.SubmodelElementCollection, aas_types.SubmodelElementList)):
            return {child.id_short: self._parse_element(child) for child in element.value} if element.value else {}
        if isinstance(element, aas_types.Property):
            return self._convert_value(element.value)
        return getattr(element, 'value', None)

    def get_continual_learning_params(self, aas_id_short: str, sm_id_short: str) -> dict:
        """
        Navigates the AAS to find and extract the parameters specifically for
        the Continual Learning pipeline.
        Path: (AAS: CA_Modul) -> (Submodel: OD_Module) -> ODOptimisation -> ContinualLeaning
        """
        # 1. Find the target AAS and its referenced Submodel
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
        if not target_submodel: raise ValueError(
            f"Submodel '{sm_id_short}' not found referenced in AAS '{aas_id_short}'.")

        # ================================================================= #
        # == THE FIX IS HERE                                             == #
        # ================================================================= #
        # The 'target_submodel' variable IS the OD_Module. We start searching from its elements.
        print("--> Navigating AAS path: OD_Module -> ODOptimisation -> ContinualLeaning")

        # 2. Find 'ODOptimisation' directly inside the 'OD_Module' submodel's elements.
        od_optimisation = next((el for el in target_submodel.submodel_elements if el.id_short == "ODOptimisation"),
                               None)
        if not od_optimisation: raise ValueError("Could not find 'ODOptimisation' element in the 'OD_Module' submodel.")

        # 3. Find 'ContinualLearning' inside 'ODOptimisation'.
        continual_learning_list = next((el for el in od_optimisation.value if el.id_short == "ContinualLeaning"), None)
        if not continual_learning_list: raise ValueError(
            "Could not find 'ContinualLeaning' element inside 'ODOptimisation'.")

        # 4. Parse the properties into a flat dictionary
        params_raw = self._parse_element(continual_learning_list)
        print("✅ Successfully extracted raw parameters from AAS.")

        # 5. Clean and validate the extracted parameters
        cl_params = {}
        try:
            cl_params['model_weights'] = str(params_raw['input_model']).strip('"')
            cl_params['current_classes'] = str(params_raw['current_classes']).strip('"')
            cl_params['new_classes'] = str(params_raw['new_classes']).strip('"')
            cl_params['replay_data_dir'] = str(params_raw['replay_data_dir']).strip('"')
            cl_params['new_data_dir'] = str(params_raw['new_data_dir']).strip('"')
            cl_params['exp_name'] = str(params_raw['exp_name'])
            cl_params['replay_ratio'] = float(params_raw['replay_ratio'])

            if 'val_replay_data_dir' in params_raw: cl_params['val_replay_data_dir'] = str(
                params_raw['val_replay_data_dir']).strip('"')
            if 'val_new_data_dir' in params_raw: cl_params['val_new_data_dir'] = str(
                params_raw['val_new_data_dir']).strip('"')
            if 'output_dir' in params_raw: cl_params['output_dir'] = str(params_raw['output_dir']).strip('"')

            cl_params['epochs'] = int(params_raw.get('epochs', 100))
            cl_params['batch_size'] = int(params_raw.get('batch_size', 16))
            cl_params['img_size'] = int(params_raw.get('img_size', 640))

        except KeyError as e:
            raise RuntimeError(f"A required parameter is missing from the AAS JSON for Continual Learning: {e}.")
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"Could not convert a CL parameter to its correct type: {e}")

        print("✅ Parameters cleaned and validated.")
        return cl_params