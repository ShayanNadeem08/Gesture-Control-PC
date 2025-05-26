"""Live Prediction with Background Blurring Preprocessing"""
import time
import cv2
import numpy as np
import torch
from torch import cat as torch_cat, device as torch_device, cuda as torch_cuda, load as torch_load, stack as torch_stack, no_grad as torch_no_grad, max as torch_max
import torch.nn as nn
from torchvision.transforms import Normalize, ToTensor, Resize, Compose
from collections import deque
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import pyautogui

def get_trace_frame(frame, hand_landmarks):
    trace_frame = torch.zeros(64,64)
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
        
        self.device = torch_device("cuda" if torch_cuda.is_available() else "cpu")
        
        # Gesture classes
        self.gesture_classes = ['down', 'left', 'right', 'up', "none"]
        
        # Initialize variables for recording and processing frames
        self.frame_count = frame_count
        self.collection_time = collection_time  # Time in seconds to collect frames
        self.frame_buffer = deque(maxlen=frame_count)
        self.all_processed_frames = []  # Store all processed frames during collection
        self.is_collecting = False
        self.hand_detected_count = 0
        self.min_hand_detected = 6  # Minimum number of frames that must have a hand
        self.collection_start_time = 0
        self.collection_end_time = 0
        
        # Define the same transformation pipeline as in training
        self.transform = Compose([
            Resize((64, 64)),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5])
        ])
        
        # FPS calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.display_fps = display_fps
        
        # Create directory for saving processed frames
        self.frames_dir = "../output/processed_frames"
        os.makedirs(self.frames_dir, exist_ok=True)

        # Create directory for debug frames (to visualize preprocessing steps)
        self.debug_dir = "../output/debug_frames"
        os.makedirs(self.debug_dir, exist_ok=True)
                
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
        trace_frame = get_trace_frame(frame, hand_landmarks)
        
        return trace_frame, marked_frame, trace_frame.numpy()
    
    def save_processed_frames(self):
        """Save the processed frames for visualization"""
        # Clear previous frames
        for file in os.listdir(self.frames_dir):
            os.remove(os.path.join(self.frames_dir, file))
        
        # Save current frame buffer
        for i, frame_tensor in enumerate(self.frame_buffer):
            # Convert tensor to numpy array for visualization
            frame_np = frame_tensor.cpu().numpy()[0]  # Get the first channel
            frame_np = (frame_np * 0.5 + 0.5) * 255  # Denormalize
            frame_np = frame_np.astype(np.uint8)
            
            # Save the frame
            cv2.imwrite(os.path.join(self.frames_dir, f"frame_{i+1}.jpg"), frame_np)
    
    def visualize_processed_frames(self, prediction, confidence):
        """Plot frames used for prediction"""
        plt.figure(figsize=(16, 8))
        for i, frame_path in enumerate(sorted(os.listdir(self.frames_dir))):
            plt.subplot(2, 4, i+1)
            frame = cv2.imread(os.path.join(self.frames_dir, frame_path), cv2.IMREAD_GRAYSCALE)
            plt.imshow(frame, cmap='gray')
            plt.title(f"Frame {i+1}")
            plt.axis('off')
        
        plt.suptitle(f"Prediction: {prediction.upper()} (Confidence: {confidence:.2f})", fontsize=16)
        plt.tight_layout()
        plt.savefig("../results/prediction_frames.png")
        plt.close()
    
    def select_evenly_spaced_frames(self):
        """Select evenly spaced frames from all collected frames"""
        total_frames = len(self.all_processed_frames)
        if total_frames <= self.frame_count:
            return self.all_processed_frames
        
        # Calculate indices for evenly spaced frames
        indices = np.linspace(0, total_frames - 1, self.frame_count, dtype=int)
        selected_frames = [self.all_processed_frames[i] for i in indices]
        return selected_frames
    
    def run(self):
        """Run the gesture recognition system"""
        cap = cv2.VideoCapture(0)

        # Status variables
        collecting_status = "Press 'c' to start collecting frames"
        prediction_result = "No prediction yet"
        confidence = 0.0

        self.all_processed_frames = []
        self.frame_buffer.clear()
        self.hand_detected_count = 0
        self.collection_start_time = time.time()
        self.is_collecting = True
        collecting_status = "Collecting started..."
        
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
                        x, y = landmark.x, landmark.y
                        res.append([x,y])
                    for landmark in prev_hand_landmarks.landmark:
                        x, y = landmark.x, landmark.y
                        prev_res.append([x,y])
                    
                    res = 64*np.array(res)
                    prev_res = 64*np.array(prev_res)
                    
                    dresults = res-prev_res
                    dX, dY = np.mean(dresults,0)
                    
                    confidence = 0
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
            if confidence > 0:
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # If collecting frames
            preprocessed_display = None
            if self.is_collecting:
                current_time = time.time()
                elapsed_time = current_time - self.collection_start_time
                
                # Add delay in frame capture
                # time.sleep(1/12)
                
                # Process the current frame with background blurring
                processed_frame, marked_frame, blurred_hand = self.preprocess_hand_frame(frame, results)
                
                if processed_frame is not None:
                    self.all_processed_frames.append(processed_frame)
                    self.hand_detected_count += 1
                    if marked_frame is not None:
                        frame = marked_frame
                    if blurred_hand is not None:
                        preprocessed_display = blurred_hand
                
                # Update collection status with time remaining
                time_left = max(0, self.collection_time - elapsed_time)
                collecting_status = f""
                
                # If collection time is over
                if True or elapsed_time >= self.collection_time:
                                        
                    # Select evenly spaced frames from all collected
                    selected_frames = self.select_evenly_spaced_frames()
                    
                    # Update frame buffer with selected frames
                    self.frame_buffer = deque(selected_frames, maxlen=self.frame_count)
                    
                    # Make prediction
                    if self.action_function != None:
                        self.action_function.put([prediction_result, confidence])

                    # Start collecting frames
                    self.all_processed_frames = []
                    self.frame_buffer.clear()
                    self.hand_detected_count = 0
                    self.collection_start_time = time.time()
                    # collecting_status = "Collecting started..."
            
            # Display the resulting frame
            cv2.imshow('Hand Gesture Recognition', frame)
            
            # Display preprocessed frame if available
            if preprocessed_display is not None:
                cv2.imshow('Preprocessed Frame', preprocessed_display)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Start collecting frames
                self.is_collecting = True
                self.all_processed_frames = []
                self.frame_buffer.clear()
                self.hand_detected_count = 0
                self.collection_start_time = time.time()
        
        # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()

if __name__=="main":
    # Initialize and run the gesture recognition system
    gesture_recognition = GestureRecognition("")
    gesture_recognition.run()