# Standard library imports
import re
from pathlib import Path
from typing import Optional, List

# Third-party imports
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

def normalize_to_neg_one_to_one(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor from [0, 1] to [-1, 1]
    
    Args:
        x (torch.Tensor): Input tensor with values in range [0, 1]
        
    Returns:
        torch.Tensor: Normalized tensor with values in range [-1, 1]
    """
    return 2 * x - 1

class DataTransforms:
    @staticmethod
    def get_transforms() -> transforms.Compose:
        """Returns the standard transformation pipeline for the datasets.
        
        Returns:
            transforms.Compose: Composed transformation pipeline
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(normalize_to_neg_one_to_one)  # Normalize to [-1, 1]
        ])

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, target_size: tuple[int, int], return_label: bool = False):
        self.data_dir = data_dir
        self.dtypes = {'uint8': 255, 'uint16': 65535, 'float32': 1.0, 'float64': 1.0}
        self.target_size = target_size
        self.return_label = return_label

        self.img_fps: list[Path] = list(data_dir.glob("[0-0][0-9]/*.tif"))
        self.transform = DataTransforms.get_transforms()

    def __len__(self):
        return len(self.img_fps)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        img_fp = self.img_fps[idx]
        image = Image.open(img_fp).convert('L')  # Open as grayscale
        image = image.resize((self.target_size[1], self.target_size[0]))
        # Convert PIL Image to numpy array to handle dtype
        image = np.array(image)

        if image.max() > 1:
            image = image / self.dtypes[str(image.dtype)]

        if self.transform:
            # Convert back to PIL Image for torchvision transforms
            image = Image.fromarray(image)
            image = self.transform(image)

        if not self.return_label:
            return image, torch.tensor([])  # Return empty tensor instead of None

        label_fp = img_fp.parent.with_name(img_fp.parent.name + "_GT") / "SEG" / (f"man_seg{img_fp.name[1:]}")
        label = Image.open(label_fp).convert('L')
        label = transforms.Resize((self.target_size[1], self.target_size[0]))(label)  # Resize label to match image
        label = transforms.ToTensor()(label).float()

        return image[None], label

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, target_size: tuple[int, int], frames_per_video: int, return_label: bool = False):
        self.data_dir = data_dir
        self.dtypes = {'uint8': 255, 'uint16': 65535, 'float32': 1.0, 'float64': 1.0}
        self.frames_per_video = frames_per_video
        self.target_size = target_size
        self.return_label = return_label
 
        self.transform = DataTransforms.get_transforms()
        self.video_folders = [folder for folder in self.data_dir.iterdir() if re.search('\d\d$',folder.name)]  # or whatever format
        self.all_video_files = list(self.data_dir.glob("[0-9][0-9]/*.tif"))

    def __len__(self):
        return len(self.all_video_files)

    def __getitem__(self, idx):
        video_file = self.all_video_files[idx]
        video_files = sorted(list(video_file.parent.glob("*.tif")))
        index = video_files.index(video_file)

        if index + self.frames_per_video > len(video_files):
            index = len(video_files) - self.frames_per_video

        video_files_batch = video_files[index:index+self.frames_per_video]

        images = torch.zeros((1, self.frames_per_video, self.target_size[0], self.target_size[1]))
        labels = torch.zeros((self.frames_per_video, self.target_size[0], self.target_size[1])) if self.return_label else torch.tensor([])

        for i, video_file in enumerate(video_files_batch):
            image = Image.open(video_file).convert('L')
            image = image.resize((self.target_size[1], self.target_size[0]))
            image = np.array(image)

            if image.max() > 1:
                image = image / self.dtypes[str(image.dtype)]

            if self.transform:
                image = Image.fromarray(image)
                image = self.transform(image)

            images[0,i] = image

            if self.return_label:
                label_fp = video_file.parents[1].with_name(video_file.parents[1].name + "_GT") / "SEG" / (f"man_seg{video_file.name[1:]}")
                label = Image.open(label_fp).convert('L')
                label = transforms.Resize((self.target_size[1], self.target_size[0]))(label)  # Resize label to match image
                label = transforms.ToTensor()(label).float()
                labels[i] = label

        return images, labels

if __name__ == "__main__":
    # Example usage
    DATA_TYPE = "video" # 'image' or 'video'
    DATASET = 'moma' 
    TARGET_SIZE = (256, 32)
    FRAMES_PER_VIDEO = 16
    DATA_DIR = Path(f"./data/{DATASET}")
    if DATA_TYPE == "image":
        dataset = ImageDataset(DATA_DIR, TARGET_SIZE)
    elif DATA_TYPE == "video":
        dataset = VideoDataset(DATA_DIR, TARGET_SIZE, FRAMES_PER_VIDEO)
    else:
        raise ValueError(f"Invalid data type: {DATA_TYPE}")

    train_loader = DataLoader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            pin_memory=True  # Optimizes data transfer to GPU
        )

    # Check dataset loading
    for images, labels in train_loader:
        print(f"Loaded batch of size {images.shape} for {DATA_TYPE}")
        break