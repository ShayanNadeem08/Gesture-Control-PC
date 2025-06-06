
from queue import Queue
from threading import Thread
import json

# Import from current files
from hand_gesture_model_mp_rule import GestureRecognition
from key_map import sendKeyPress

# Read settings file
f = open("../app/config.json")
config = json.load(f)
f.close()

# Establish communcation between threads using que
gesture_que = Queue()
comm_que = Queue()

# Initialize and run the gesture recognition system
gesture_recognition = GestureRecognition("", action_function=gesture_que)

# Run threads
thread1 = Thread(target=gesture_recognition.run)
thread2 = Thread(target=sendKeyPress, args=(gesture_que, comm_que, config))

thread1.start()
thread2.start()
thread1.join()
comm_que.put("END")
thread2.join()

