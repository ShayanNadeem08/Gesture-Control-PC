import pyautogui
import time

gesture_to_key = {
    'up'  : 'up',     # Jump
    'down': 'down',   # Slide
    'left': 'left',   # Move left
    'right':'right'   # Move right
}

def sendKeyPress(gesture_que, comm_que):
    """ This function performs action when a prediction is made."""

    while True:
        time.sleep(0.125)
        try:
            command = comm_que.get(False)
            if command == "END": break
        except:
            pass

        try:
            prediction, confidence = gesture_que.get_nowait()
            
            if prediction != "Insufficient frames":
                if confidence > 0.9:
                    print(gesture_to_key[prediction])
                    pyautogui.press(gesture_to_key[prediction])
        except:
            pass
    print("E")