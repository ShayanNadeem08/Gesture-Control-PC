
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

from global_defines import GESTURE_MAP_STR2N as gesture_map

# Custom Dataset class for hand gestures
class HandGestureDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# Function to load and preprocess the dataset
def load_dataset(data_path):
    """
    Load and preprocess the hand gesture dataset.

    Args:
        data_path: Path to the dataset directory

    Returns:
        X: Sequences of images with shape (num_sequences, channels, frames, height, width)
        y: Labels corresponding to gestures (0: down, 1: left, 2: right, 3: up)
        subject_ids: IDs of subjects for leave-one-subject-out cross-validation
    """
    # Placeholder arrays for data, labels, and subject IDs
    sequences = []
    labels = []
    subject_ids = []

    

    # Image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Using power of 2 dimensions for better downsampling
        transforms.ToTensor(),  # Converts to [0,1] range and changes to CxHxW format
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1] range
    ])

    # Iterate through gesture folders
    for gesture in ['down', 'left', 'right', 'up']:
        gesture_path = os.path.join(data_path, gesture)

        # Iterate through sequence folders for this gesture
        for seq_folder in os.listdir(gesture_path):
            seq_path = os.path.join(gesture_path, seq_folder)

            if os.path.isdir(seq_path):
                # Extract subject ID from sequence folder name (assuming format contains subject ID)
                # Modify this according to your actual folder naming convention
                subject_id = int(seq_folder.split('_')[0])

                # Load frames for this sequence
                frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.jpg') or f.endswith('.png')])

                if len(frame_files) > 0:
                    # Load and normalize frames
                    frames = []
                    for frame_file in frame_files:
                        frame_path = os.path.join(seq_path, frame_file)
                        # Open image and apply transformations
                        img = Image.open(frame_path).convert('L')  # Convert to grayscale
                        img_tensor = transform(img)  # Apply transforms
                        frames.append(img_tensor)

                    # Pad or truncate sequence to fixed length (8 frames - power of 2)
                    target_frames = 8
                    if len(frames) < target_frames:
                        # Pad with zeros if sequence is too short
                        for _ in range(target_frames - len(frames)):
                            frames.append(torch.zeros_like(frames[0]))
                    elif len(frames) > target_frames:
                        # Truncate if sequence is too long
                        frames = frames[:target_frames]

                    # Stack frames along a new dimension
                    sequence_tensor = torch.stack(frames)  # Shape: [frames, channels, height, width]
                    sequence_tensor = sequence_tensor.permute(1, 0, 2, 3)  # Reshape to [channels, frames, height, width]

                    # Add sequence to dataset
                    sequences.append(sequence_tensor)
                    labels.append(gesture_map[gesture])
                    subject_ids.append(subject_id)

    # Convert lists to tensors
    X = torch.stack(sequences)
    y = torch.tensor(labels, dtype=torch.long)
    subject_ids = np.array(subject_ids)

    return X, y, subject_ids