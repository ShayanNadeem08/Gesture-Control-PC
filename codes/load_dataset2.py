
from torch.utils.data import DataLoader, random_split
import sys
sys.path.insert(1, "../external_modules/")
from video_dataset import VideoFrameDataset, ImglistToTensor

# Data transforms
data_transforms = transforms.Compose([
                    ImglistToTensor()
                    transforms.Resize((64, 64)),  # Using power of 2 dimensions for better downsampling
                    transforms.ToTensor(),  # Converts to [0,1] range and changes to CxHxW format
                    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1] range
                  ])

# Load dataset
def load_dataset(dataset_path, *split_ratios, batch_size):

    dataset = VideoFrameDataset(
        root_path= f"{dataset_path}",
        annotationfile_path=f"{dataset_path}/../annotations.txt",
        num_segments=8,
        frames_per_segment=1,
        imagefile_template='{:01d}.jpg',
        transform=data_transforms,
        test_mode=False
    )
    
    train_dataset, valid_dataset, test_dataset = random_split(dataset, *split_ratios)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader
