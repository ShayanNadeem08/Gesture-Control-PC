
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from collections import deque
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image
import time
import os

class GestureRecognition:
    def __init__(self, model, action_function, frame_count=8, collection_time=1.0, display_fps=True):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Load the trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.action_function = action_function

        # Gesture classes
        self.gesture_classes = ['down', 'left', 'right', 'up']

        # Initialize variables for frames processing
        self.frame_count = frame_count
        self.collection_time = collection_time  # Time in seconds to collect frames
        self.frame_buffer = deque(maxlen=frame_count)
        self.all_processed_frames = []  # Store all processed frames during collection
        self.is_collecting = False
        self.hand_detected_count = 0
        self.min_hand_detected = 6  # Minimum number of frames that must have a hand

        # Collection timing variables
        self.collection_start_time = 0
        self.collection_end_time = 0

        # Define the same transformation pipeline as in training
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # FPS calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.display_fps = display_fps

        # Create directory for saving processed frames
        self.frames_dir = "../results/processed_frames"
        os.makedirs(self.frames_dir, exist_ok=True)

    def preprocess_hand_frame(self, frame, results):
        """Extract hand from frame and preprocess it"""
        h, w, _ = frame.shape

        # If no hand is detected, return None
        if not results.multi_hand_landmarks:
            return None, None

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
            return None, None

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

        # Convert to grayscale and PIL image
        hand_crop_gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(hand_crop_gray)

        # Apply transforms
        img_tensor = self.transform(pil_img)

        return img_tensor, marked_frame

    def predict_gesture(self):
        """Make a prediction based on collected frames"""
        if len(self.frame_buffer) < self.frame_count:
            return "Insufficient frames", 0.0

        if self.hand_detected_count < self.min_hand_detected:
            return "Hand not consistently detected", 0.0

        # Stack frames into a sequence
        sequence = torch.stack(list(self.frame_buffer))  # Shape: [frames, channels, height, width]
        sequence = sequence.permute(1, 0, 2, 3).unsqueeze(0)  # Reshape to [1, channels, frames, height, width]

        # Make prediction
        with torch.no_grad():
            sequence = sequence.to(self.device)
            outputs = self.model(sequence)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)

        # Get prediction class and confidence
        predicted_class = self.gesture_classes[prediction.item()]
        confidence_value = confidence.item()

        return predicted_class, confidence_value

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

            # Draw hand landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Display collection status
            cv2.putText(frame, collecting_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display prediction result
            cv2.putText(frame, f"Prediction: {prediction_result}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if confidence > 0:
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # If collecting frames
            if self.is_collecting:
                current_time = time.time()
                elapsed_time = current_time - self.collection_start_time

                # Process the current frame
                processed_frame, marked_frame = self.preprocess_hand_frame(frame, results)

                if processed_frame is not None:
                    self.all_processed_frames.append(processed_frame)
                    self.hand_detected_count += 1
                    if marked_frame is not None:
                        frame = marked_frame

                # Update collection status with time remaining
                time_left = max(0, self.collection_time - elapsed_time)
                collecting_status = f"Collecting: {elapsed_time:.1f}s / {self.collection_time:.1f}s"

                # Add a progress bar
                progress = int(min(elapsed_time / self.collection_time, 1.0) * 200)
                cv2.rectangle(frame, (10, 150), (210, 170), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 150), (10 + progress, 170), (0, 255, 0), -1)

                # If collection time is over
                if elapsed_time >= self.collection_time:
                    self.is_collecting = False

                    # Select evenly spaced frames from all collected
                    selected_frames = self.select_evenly_spaced_frames()

                    # Update frame buffer with selected frames
                    self.frame_buffer = deque(selected_frames, maxlen=self.frame_count)

                    # Make prediction
                    prediction_result, confidence = self.predict_gesture()
                    collecting_status = "Press 'c' to collect new frames"

                    # Take action based on prediction
                    self.action_function(prediction_result, confidence)

                    # Save and visualize processed frames
                    self.save_processed_frames()
                    self.visualize_processed_frames(prediction_result, confidence)

                    # Wait for some time
#                    time.sleep(1/100)

                    # Start collecting frames
                    self.is_collecting = True
                    self.all_processed_frames = []
                    self.frame_buffer.clear()
                    self.hand_detected_count = 0
                    self.collection_start_time = time.time()
                    collecting_status = "Collecting started..."


            # Display the resulting frame
            cv2.imshow('Hand Gesture Recognition', frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()