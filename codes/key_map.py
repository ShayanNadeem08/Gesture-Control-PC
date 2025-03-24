import pyautogui

gesture_to_key = {
    'up'  : 'up',     # Jump
    'down': 'down',   # Slide
    'left': 'left',   # Move left
    'right':'right'   # Move right
}

def sendKeyPress(prediction, confidence):
    """ This function is called when a prediction is made."""
    if prediction != "Insufficient frames":
        pyautogui.press(gesture_to_key[prediction])