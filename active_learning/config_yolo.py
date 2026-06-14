# config_yolo.py

# --- Project & Path Configuration ---
PROJECT_NAME = "yolo_al_experiment" # A name for the experiment run
DATA_DIR = "./data" # Root directory for all data
OUTPUT_DIR = "./output" # Directory for all models and results

# Original full dataset
F_FULL_DATA = "full-data"
F_TRAIN = "train"
F_VAL = "val"

# Directory to store the active learning pool and round-specific data
F_WORKDIR = "workspace"
F_UNLABELED = "unlabeled"

# File extensions
IMG_EXTENSION = ".jpg"
ANN_EXTENSION = ".txt"

# --- YOLO Model & Training Configuration ---
INITIAL_MODEL = "../OD_training/yolov12l.pt" # Pretrained model to start from (YOLOv12, project default)
EPOCHS_PER_ROUND = 5 # Number of epochs to train in each AL round
BATCH_SIZE = 16
IMG_SIZE = 640
NUM_CLASSES = 8 # IMPORTANT: Change this to your dataset's class count
DATASET_YAML = f"{DATA_DIR}/{F_FULL_DATA}/dataset.yaml" # Path to your dataset's YAML file

# --- Active Learning Configuration ---
AL_ROUNDS = 5 # Total number of active learning rounds
INITIAL_SUBSET_SIZE = 10 # Number of images to start with for the first training round
ADD_PER_ROUND = 5 # Number of new images to select and add in each AL round

# Split ratio for Uncertainty vs. Diversity. 1.0 = all Uncertainty, 0.0 = all Diversity.
# This can be a list to vary the ratio per round, e.g., [1.0, 0.75, 0.5, 0.25]
UD_SPLIT = 0.5

# Confidence threshold for the model when performing inference for AL selection
AL_SCORE_THRESHOLD = 0.10

# --- Reproducibility ---
SEED = 42