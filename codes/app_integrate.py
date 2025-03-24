import torch

# Import from current files
from global_defines import *
from model import DenseNet3D
from gesture_recog import GestureRecognition
from key_map import sendKeyPress

# Load model
model = torch.load(MODEL_PATH+"/DENSENET.model")
print("Model loaded")

# Initialize and run the gesture recognition system
model = DenseNet3D(growth_rate=12, block_config=(2, 4, 4), num__init__features=16)
model.load_state_dict(torch.load(MODEL_PATH+"/DENSENET.model"))

# Run and predict
gesture_recognition = GestureRecognition(model, sendKeyPress)
gesture_recognition.run()