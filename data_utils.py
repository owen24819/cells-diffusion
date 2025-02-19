import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path
import numpy as np
from typing import Optional

def normalize_to_neg_one_to_one(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor from [0, 1] to [-1, 1]
    
    Args:
        x (torch.Tensor): Input tensor with values in range [0, 1]
        
    Returns:
        torch.Tensor: Normalized tensor with values in range [-1, 1]
    """
    return 2 * x - 1

def load_dataset(data_dir: Path) -> torch.utils.data.Dataset:
    """
    Load the specified dataset.
    
    Args:
        data_dir (str): Directory to store/download the dataset.
    
    Returns:
        torch.utils.data.Dataset: The requested dataset.
    """

    dataset_name = data_dir.name

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(normalize_to_neg_one_to_one)  # Normalize to [-1, 1]
    ])
    
    if dataset_name == "cells":
        return cells_dataset(data_dir, transform, target_size=(256, 32))
    
    try:
        dataset_class = getattr(datasets, dataset_name)
        download = False if (data_dir).exists() else True
        torch_dataset = dataset_class(root=data_dir.parent, train=True, download=download, transform=transform)
        image = torch_dataset[0][0]
        torch_dataset.target_size = (image.shape[1], image.shape[2])
        return torch_dataset
    except AttributeError:
        example_datasets = ["MNIST", "CIFAR10", "FashionMNIST", "cells"]
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Try one of these examples: {', '.join(example_datasets)}")

class cells_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: Path = Path("./data/cells"),
        transform: Optional[transforms.Compose] = None,
        target_size: tuple[int, int] = (256, 32),
        return_label: bool = True
    ):

        self.data_dir: Path = data_dir
        self.transform: Optional[transforms.Compose] = transform
        self.target_size: tuple[int, int] = target_size
        self.return_label: bool = return_label

        self.img_fps: list[Path] = list(data_dir.glob("[0-0][0-9]/*.tif"))
        self.dtypes: dict[str, float] = {'uint8': 255, 'uint16': 65535, 'float32': 1.0, 'float64': 1.0}

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

        label: Optional[torch.Tensor] = None
        if self.return_label:
            label_fp = img_fp.parent.with_name(img_fp.parent.name + "_GT") / "SEG" / (f"man_seg{img_fp.name[1:]}")
            label = Image.open(label_fp).convert('L')
            label = transforms.Resize((256, 256))(label)  # Resize label to match image
            label = transforms.ToTensor()(label).float()

        return image, label

if __name__ == "__main__":
    # Example usage
    DATASET = "MNIST"  # Change this to load a different dataset
    dataset = load_dataset(data_dir=Path(f"./data/{DATASET}"))

    train_loader = DataLoader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            pin_memory=True  # Optimizes data transfer to GPU
        )

    # Check dataset loading
    for images, labels in train_loader:
        print(f"Loaded batch of size {images.shape}")
        break