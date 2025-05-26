import pyautogui
import time

# Gesture-to-key mapping for Subway Surfers
gesture_to_key = {
    0: 'up',     # Jump
    1: 'down',   # Slide
    2: 'left',   # Move left
    3: 'right'   # Move right
}

def map_gesture_to_game_action(gesture_number):
    """
    Mapping the given gesture number to the corresponding keypress in the game.
    """
    key_to_press = gesture_to_key.get(gesture_number, None)
    if key_to_press:
        pyautogui.press(key_to_press)
        print(f"Gesture {gesture_number} detected. Key pressed: {key_to_press}")
    else:
        print(f"Invalid gesture: {gesture_number}")

try:
    while True:
        # input should be the gesture that has detected
        gesture_input = int()
        map_gesture_to_game_action(gesture_input)
        time.sleep(0.5)  # Adding a slight delay to simulate real gameplay
except KeyboardInterrupt:
    print("Exiting simulation.")