import pyautogui

gesture_to_key = {
    'up'  : 'up',     # Jump
    'down': 'down',   # Slide
    'left': 'left',   # Move left
    'right':'right'   # Move right
}

def sendKeyPress(gesture_que, comm_que):
    """ This function performs action when a prediction is made."""
    while True:
        try:
            command = comm_que.get_nowait()
            if command == "END": return
        except queue.Empty:
            pass

        try:
            prediction, confidence = gesture_que.get_nowait()
        except:
            continue

        if prediction != "Insufficient frames":
            if confidence > 0.9:
                print(gesture_to_key[prediction])
                pyautogui.press(gesture_to_key[prediction])