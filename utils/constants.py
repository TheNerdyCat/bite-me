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
MODEL_NAME = "xception"

# Epochs
PATIENCE = 11
MIN_EPOCHS = 30
MAX_EPOCHS = 50

# Optimizer & learning rate constants
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0
EPSILON = 1e-8 #1e-08
AMSGRAD = True
BETAS = (0.9, 0.999)
ETA_MIN = 0.00005 #0.0001

PRECISION = 16
GRADIENT_CLIP_VAL = 1.0

LOG_DIR = "../logs/logs"
LOG_NAME = str(datetime.datetime.now()).replace("-", "_").replace(" ", "_")[:19]
