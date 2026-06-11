from aas_config_converter import AASConfigConverter
import yaml

# Path to the Asset Administration Shell file
AAS_FILE_PATH = "../../AAS/CA_Modul-OD_pipeline_v0.1.json"

# Initialize the converter
config_provider = AASConfigConverter(AAS_FILE_PATH)

# Get the configuration dictionary
sdg_params = config_provider.get_config_from_submodel(
    aas_id_short="CA_Modul",
    sm_id_short="OD_Module"
)

# Now you can access parameters directly from the dictionary
print("Loading configuration for SDG...")
# The first key is 'SmartSDG', so we access its content
smart_sdg_config = sdg_params.get("SmartSDG", {})

base_path = smart_sdg_config.get("Paths", {}).get("base")
img_width = smart_sdg_config.get("generation_params", {}).get("Camera", {}).get("img_width")

print(f"Base asset path: {base_path}")
print(f"Image width: {img_width}")

# Your SDG logic continues here, using the 'smart_sdg_config' dictionary...