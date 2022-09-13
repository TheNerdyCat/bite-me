import torch
import datetime

# Set img dimension constants
ROWS = 1024
COLS = 1024
CHANNELS = 3

# Set seed
SEED = 14

# Device constants
DEVICE_NAME = torch.cuda.get_device_name(0)
N_WORKERS = 128

# Set test size
TEST_SIZE = 0.1

# Set number of splits for cross-validation
N_SPLITS = 3

# Batch sizes
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4

# Verbosity
VERBOSE = True

# Choose model
MODEL_NAME = "resnet152"

# Number of epochs
MIN_EPOCHS = 50
MAX_EPOCHS = 70

PATIENCE = 11

PRECISION = 16
GRADIENT_CLIP_VAL = 1.0

LOG_DIR = "../logs/logs"
LOG_NAME = str(datetime.datetime.now()).replace("-", "_").replace(" ", "_")[:19]
