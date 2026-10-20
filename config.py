import os

# ---- Paths ----
ROOT_DIR = "./monai_dataset"
MODEL_SAVE_PATH =  "/com_neuro_project/weights/segmentation_model.pth"
TASK = "Task01_BrainTumour"

# ---- Data ----
SPATIAL_SIZE = (96, 96, 96)
NUM_SAMPLES = 2          # random crops per volume
BATCH_SIZE = 2
VAL_BATCH_SIZE = 1
NUM_WORKERS = 2
CACHE_RATE = 0

# ---- Model ----
IN_CHANNELS = 4          # 4 MRI modalities
OUT_CHANNELS = 4         # bg + 3 tumor subregions
CHANNELS = (16, 32, 64, 128, 256)
STRIDES = (2, 2, 2, 2)
NUM_RES_UNITS = 2

# ---- Training ----
NUM_EPOCHS = 15
LR = 1e-4
DEVICE = "cuda"          # resolved at runtime in train.py

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
