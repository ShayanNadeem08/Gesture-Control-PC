"""Live Prediction with Background Blurring Preprocessing"""
import cv2
import numpy as np
import mediapipe as mp
import time

GESTURE_MAP_STR2NUM = {'down': 0, 'left': 1, 'right': 2, 'up': 3, "none":4}
GESTURE_MAP_NUM2STR = {0: 'down', 1: 'left', 2: 'right', 3: 'up', 4:"none"}

class GestureRecognition:
    def __init__(self, model_path, frame_count=8, collection_time=1.0, display_fps=True, action_function=None):
        self.action_function = action_function

        # Load MediaPipe Hands model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Gesture classes
        self.gesture_classes = ['down', 'left', 'right', 'up', "none"]
        
        # Initialize variables for recording and processing frames
        
        # FPS calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.display_fps = display_fps

    def get_trace_frame(self, frame, hand_landmarks):
        trace_frame = np.zeros((64,64))
        for landmark in hand_landmarks.landmark:
            y = int(63*landmark.x)
            x = int(63*landmark.y)
            
            N = 63
            X = 1
            trace_frame[min(x,N)][min(y,N)] = X
            
            trace_frame[min(x+1,N)][min(y,N)] = X
            trace_frame[min(max(x-1,0),N)][min(y,N)] = X
            trace_frame[min(x,N)][min(y+1,N)] = X
            trace_frame[min(x,N)][min(max(y-1,0),N)] = X
            
            trace_frame[min(x+1,N)][min(y+1,N)] = X
            trace_frame[min(max(x-1,0),N)][min(y+1,N)] = X
            trace_frame[min(x+1,N)][min(max(y-1,0),N)] = X
            trace_frame[min(max(x-1,0),N)][min(max(y-1,0),N)] = X

        return trace_frame
                
    def preprocess_hand_frame(self, frame, results):
        """Extract hand from frame and preprocess it with background blurring"""
        h, w, _ = frame.shape
        
        # If no hand is detected, return None
        if not results.multi_hand_landmarks:
            return None, None, None
        
        # Get landmarks for the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Calculate bounding box with padding
        x_min, x_max, y_min, y_max = w, 0, h, 0
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Check if bounding box is valid
        if x_min >= x_max or y_min >= y_max:
            return None, None, None
        
        # Crop the hand region
        hand_crop = frame[y_min:y_max, x_min:x_max].copy()
        
        # Create a copy of the original frame with hand landmarks
        marked_frame = frame.copy()
        self.mp_drawing.draw_landmarks(
            marked_frame, 
            hand_landmarks, 
            self.mp_hands.HAND_CONNECTIONS
        )
        
        # Draw bounding box
        cv2.rectangle(marked_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Apply skin detection and background blurring
        trace_frame = self.get_trace_frame(frame, hand_landmarks)
        
        return trace_frame, marked_frame, trace_frame
    
    def run(self):
        """Run the gesture recognition system"""
        cap = cv2.VideoCapture(0)

        # Status variables
        collecting_status = "Collecting frames"
        prediction_result = "No prediction yet"
        
        # Create a window to display the preprocessed frames
        cv2.namedWindow('Preprocessed Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Preprocessed Frame', 320, 240)
        
        prev_results = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            if self.display_fps:
                self.new_frame_time = time.time()
                fps = 1/(self.new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
                self.prev_frame_time = self.new_frame_time
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Make prediction
            if results.multi_hand_landmarks:
                if not prev_results:
                    prev_results = results
                else:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    prev_hand_landmarks = prev_results.multi_hand_landmarks[0]
                    prev_results = results

                    # Calculate bounding box with padding
                    res=[]
                    prev_res=[]
                    for landmark in hand_landmarks.landmark:
                        x, y = 64*landmark.x, 64*landmark.y
                        res.append([x,y])
                    for landmark in prev_hand_landmarks.landmark:
                        x, y = 64*landmark.x, 64*landmark.y
                        prev_res.append([x,y])
                    
                    res = np.array(res)
                    prev_res = np.array(prev_res)
                    
                    dresults = res-prev_res
                    dX, dY = np.mean(dresults,0)
                    
                    epsilon = 1
                    if abs(dX) < epsilon and abs(dY) < epsilon:
                        prediction_result = "none"
                    elif abs(dX) > abs(dY):
                        if dX > 0:
                            prediction_result="left"
                        else:
                            prediction_result="right"
                    elif abs(dX) < abs(dY):
                        if dY > 0:
                            prediction_result="down"
                        else:
                            prediction_result="up"
            else:
                prediction_result = "none"
            
            # Display collection status
            cv2.putText(frame, collecting_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display prediction result
            cv2.putText(frame, f"Prediction: {prediction_result}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Display quit message
            cv2.putText(frame, f"Press 'q' to exit", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            preprocessed_display = None
            
            # Add delay in frame capture
            # time.sleep(1/64)
            
            # Process the current frame with background blurring
            processed_frame, marked_frame, blurred_hand = self.preprocess_hand_frame(frame, results)
            
            if processed_frame is not None:
                if marked_frame is not None:
                    frame = marked_frame
                if blurred_hand is not None:
                    preprocessed_display = blurred_hand
            
            # Send prediction to other thread
            if self.action_function != None:
                self.action_function.put(prediction_result)
            
            # Display the resulting frame
            cv2.imshow('Hand Gesture Recognition', frame)
            
            # Display preprocessed frame if available
            if preprocessed_display is not None:
                cv2.imshow('Preprocessed Frame', preprocessed_display)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()

if __name__=="main":
    # Initialize and run the gesture recognition system
    gesture_recognition = GestureRecognition("")
    gesture_recognition.run()