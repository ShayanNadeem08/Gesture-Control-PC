import pyautogui
import keyboard
import time


def sendKeyPress(gesture_que, comm_que, config):
    """ This function performs action when a prediction is made."""
    # Load key map from settings
    gesture_to_key = config["gesture_map"]

    while True:
        # Wait between key presses
        time.sleep(1/32)

        # Recieve commands from main thread
        try:
            command = comm_que.get_nowait()
            if command == "END": break
        except:
            pass
        
        # Recieve prediction from other thread
        try:
            prediction, confidence = gesture_que.get_nowait()
            
            if prediction != "Insufficient frames":
                print(gesture_to_key[prediction])
                pyautogui.press(gesture_to_key[prediction])
        except:
            pass
    print("End")