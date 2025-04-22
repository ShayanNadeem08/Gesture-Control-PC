import pyautogui
import time

# Load key map from file
gesture_to_key = {}
gesture_file = open("../app/gesture_map.txt")
gesture_to_key["down"] = gesture_file.readline()
gesture_to_key["left"] = gesture_file.readline()
gesture_to_key["right"] = gesture_file.readline()
gesture_to_key["up"] = gesture_file.readline()

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