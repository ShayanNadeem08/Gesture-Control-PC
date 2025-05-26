import pyautogui
import keyboard
import time

# Load key map from file
gesture_to_key = {}
gesture_file = open("../app/gesture_map.txt")
gesture_to_key["down"] = gesture_file.readline()[:-1]
gesture_to_key["left"] = gesture_file.readline()[:-1]
gesture_to_key["right"] = gesture_file.readline()[:-1]
gesture_to_key["up"] = gesture_file.readline()[:-1]

print(gesture_to_key)

def sendKeyPress(gesture_que, comm_que):
    """ This function performs action when a prediction is made."""

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