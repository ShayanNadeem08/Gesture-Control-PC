"""Live Prediction with Background Blurring Preprocessing"""
import time
t0 = time.time()
print(time.time()-t0); t0=time.time()
import cv2
print("cv:",time.time()-t0); t0=time.time()
import numpy as np
print("np:",time.time()-t0); t0=time.time()
import torch
from torch import cat as torch_cat, device as torch_device, cuda as torch_cuda, load as torch_load, stack as torch_stack, no_grad as torch_no_grad, max as torch_max
print("torch:",time.time()-t0); t0=time.time()
import torch.nn as nn
print("nn:",time.time()-t0); t0=time.time()
from torchvision.transforms import Normalize, ToTensor, Resize, Compose
print("transforms:",time.time()-t0); t0=time.time()
from collections import deque
print("deque:",time.time()-t0); t0=time.time()
import mediapipe as mp
print("mp:",time.time()-t0); t0=time.time()
import matplotlib.pyplot as plt
from PIL import Image
import time
import os

kdx=0
idx=0
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
    
    if False:
        global kdx, idx
        import cv2
        folder = "../../Dataset/mp_g"
        label = "none"
        start_no = 542
        if not os.path.isdir(f"{folder}/image/{label}/{start_no+kdx}/"):
            os.mkdir(f"{folder}/image/{label}/{start_no+kdx}/")
        cv2.imwrite(f"{folder}/image/{label}/{start_no+kdx}/{idx}.jpg", trace_frame.numpy()*255)
        
        idx+=1
        if idx==8:
            idx=0
            kdx+=1
    
    return trace_frame

# Recreate the DenseNet3D model to match your training architecture
class DenseBlock3D(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock3D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_dense_layer(in_channels + i * growth_rate, growth_rate))
    
    def _make_dense_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm3d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv3d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch_cat(features, 1))
            features.append(new_feature)
        return torch_cat(features, 1)


class TransitionLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer3D, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.layers(x)


class DenseNet3D(nn.Module):
    def __init__(self, growth_rate=12, block_config=(2, 4, 4), num_init_features=16, 
                 compression_rate=0.5, num_classes=5):
        super(DenseNet3D, self).__init__()
        
        # First convolution and pooling
        self.features = nn.Sequential(
            nn.Conv3d(1, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Dense Blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # Add a dense block
            block = DenseBlock3D(
                in_channels=num_features,
                growth_rate=growth_rate,
                num_layers=num_layers
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            
            # Add a transition layer (except after the last block)
            if i != len(block_config) - 1:
                out_features = int(num_features * compression_rate)
                trans = TransitionLayer3D(num_features, out_features)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = out_features
        
        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))
        self.features.add_module('relu_final', nn.ReLU(inplace=True))
        
        # Global Average Pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class GestureRecognition:
    def __init__(self, model_path, frame_count=8, collection_time=1.0, display_fps=True, action_function=None):
        self.action_function = action_function

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
        self.device = torch_device("cuda" if torch_cuda.is_available() else "cpu")
        self.model = DenseNet3D(growth_rate=12, block_config=(2, 4, 4), num_init_features=16).to(self.device)
        self.model.load_state_dict(torch_load(model_path, map_location=self.device))
        self.model.eval()
        
        # Gesture classes
        self.gesture_classes = ['down', 'left', 'right', 'up', "none"]
        
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
        self.frames_dir = "processed_frames"
        os.makedirs(self.frames_dir, exist_ok=True)

        # Create directory for debug frames (to visualize preprocessing steps)
        self.debug_dir = "debug_frames"
        os.makedirs(self.debug_dir, exist_ok=True)
        
    def detect_skin(self, img_cv):
        """
        Detect skin pixels in an image using HSV color space.
        
        Args:
            img_cv: OpenCV image in BGR format
        
        Returns:
            mask: Binary mask where skin pixels are white
        """
        # Convert to HSV color space
        img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color in HSV
        lower_skin = np.array([0, 0, 0], dtype=np.uint8)
        upper_skin = np.array([255, 255, 255], dtype=np.uint8)
        
        # Create binary mask
        mask = cv2.inRange(img_hsv, lower_skin, upper_skin)
        
        # Additional morphological operations to improve the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask

    def blur_background(self, img_cv, mask, blur_strength=21):
        """
        Blur the background of an image while keeping the hand in focus.
        
        Args:
            img_cv: OpenCV image in BGR format
            mask: Binary mask where hand pixels are white
            blur_strength: Strength of the Gaussian blur (odd number)
        
        Returns:
            result: Image with blurred background
        """
        # Ensure blur_strength is odd
        if blur_strength % 2 == 0:
            blur_strength += 1
        
        # Apply Gaussian blur to the entire image
        blurred = cv2.GaussianBlur(img_cv, (blur_strength, blur_strength), 0)
        
        # Create inverted mask for background
        mask_inv = cv2.bitwise_not(mask)
        
        # Get the background from the blurred image
        background = cv2.bitwise_and(blurred, blurred, mask=mask_inv)
        
        # Get the hand from the original image
        hand = cv2.bitwise_and(img_cv, img_cv, mask=mask)
        
        # Combine hand and blurred background
        result = cv2.add(hand, background)
        
        return result
        
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
    
    def predict_gesture(self):
        """Make a prediction based on collected frames"""
        if len(self.frame_buffer) < self.frame_count:
            return "Insufficient frames", 0.0
        
        if self.hand_detected_count < self.min_hand_detected:
            return "Hand not consistently detected", 0.0
        
        # Stack frames into a sequence
        sequence = torch_stack(list(self.frame_buffer))  # Shape: [frames, channels, height, width]
        sequence = torch.from_numpy(np.array([[sequence]]))
        # sequence = sequence.permute(1, 0, 2, 3).unsqueeze(0)  # Reshape to [1, channels, frames, height, width]
        
        # Make prediction
        with torch_no_grad():
            global M
            if M==0:
                M=1
                print(sequence.shape)
                f = open("live.txt","w")
                for frame in range(8):
                    for row in range(64):
                        for col in range(64):
                            f.write(str(int(sequence[0][0][frame][row][col]))+" ")
                        f.write("\n")
                    f.write("\n")
                f.close()

            sequence = sequence.to(self.device)
            outputs = self.model(sequence)
            probabilities = nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch_max(probabilities, 1)
            
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
        global M
        M=0
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
            # if results.multi_hand_landmarks:
            #     for hand_landmarks in results.multi_hand_landmarks:
            #         self.mp_drawing.draw_landmarks(
            #             frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
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
                time.sleep(1/12)
                
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
                collecting_status = f"Collecting: {elapsed_time:.1f}s / {self.collection_time:.1f}s"
                
                # Add a progress bar
                progress = int(min(elapsed_time / self.collection_time, 1.0) * 200)
                cv2.rectangle(frame, (10, 150), (210, 170), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 150), (10 + progress, 170), (0, 255, 0), -1)
                
                # If collection time is over
                if elapsed_time >= self.collection_time:
                                        
                    # Select evenly spaced frames from all collected
                    selected_frames = self.select_evenly_spaced_frames()
                    
                    # Update frame buffer with selected frames
                    self.frame_buffer = deque(selected_frames, maxlen=self.frame_count)
                    
                    # Make prediction
                    prediction_result, confidence = self.predict_gesture()
                    if self.action_function != None:
                        self.action_function.put([prediction_result, confidence])
                    # collecting_status = "Press 'c' to collect new frames"
                    
                    # Save and visualize processed frames
                    # self.save_processed_frames()
                    # self.visualize_processed_frames(prediction_result, confidence)

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
                collecting_status = "Collecting started..."
        
        # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()

if __name__=="main":
    # Replace with the path to your trained model
    # model_path = "../model/hand_gesture_model_blur_bg.pth"
    model_path = "../model/densenet_mediapipe_mp.pth"
    # Initialize and run the gesture recognition system
    # Set collection_time to 1.0 seconds
    gesture_recognition = GestureRecognition(model_path, collection_time=1.0)
    gesture_recognition.run()