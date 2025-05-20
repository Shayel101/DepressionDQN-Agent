import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Path to participants file (assuming CSV now)
SCORES_PATH = os.path.join(DATA_DIR, "participants.csv")

# Directories for time-series CSV files
CONDITIONS_DIR = os.path.join(DATA_DIR, "condition")
CONTROLS_DIR = os.path.join(DATA_DIR, "control")

# Sequence length for time-series
SEQUENCE_LENGTH = 100

# Other hyperparameters...
TRANSFORMER_EMBED_DIM = 64
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_FF_DIM = 128
LEARNING_RATE = 1e-3
GAMMA = 0.99
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 100
NUM_EPISODES = 1000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
NUM_CLASSES = 2  # depressed vs control
THRESHOLD = 20
