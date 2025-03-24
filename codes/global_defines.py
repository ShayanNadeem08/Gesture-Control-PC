
# Path to your dataset
DATASET_PATH = r"../../Dataset/STMM/"
MODEL_PATH = r"../model/"

# Map gesture names with numerical label
GESTURE_MAP_STR2NUM = {'down': 0, 'left': 1, 'right': 2, 'up': 3}
GESTURE_MAP_NUM2STR = {0: 'down', 1: 'left', 2: 'right', 3: 'up'}
GESTURE_CLASSES = ('down', 'left', 'right', 'up')

# Set to True to skip time taking parts of code to see if rest of it works
CHECKING_MODE=False
# Set to True to print messages
DEBUG_MODE=False