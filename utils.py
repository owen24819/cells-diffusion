import torch
from torch.utils.data import DataLoader

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List
import math
import numpy as np

from diffusers import DDPMScheduler

from data_utils import ImageDataset, VideoDataset

def add_noise(
        images: torch.Tensor, 
        timesteps: int, 
        noise_scheduler: DDPMScheduler
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Add noise to images using the DDPM scheduler.
    
    Args:
        images: Tensor of images in [-1, 1] range
        timesteps: Tensor of timesteps for each image
        device: torch device to use
        
    Returns:
        noisy_images: Images with added noise
        noise: The noise that was added
    """
    noise = torch.randn_like(images)
    noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
    return noisy_images, noise

# Custom learning rate schedule with warmup and cosine decay
def lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / warmup_steps  # Linear warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

def convert_neg_one_to_one_to_np_8bit(tensor: torch.Tensor) -> np.ndarray:
    tensor =  (tensor.squeeze() + 1) / 2
    return (tensor.detach().cpu().numpy() * 255).astype('uint8')

def save_noisy_images(
        data_dir: Path, 
        data_type: str,
        target_size: tuple[int, int],
        noise_scheduler: DDPMScheduler,
        timesteps: List[int],
        num_images_to_save: int, 
        ):
    

    if data_type == "image":
        train_dataset = ImageDataset(data_dir, target_size)
    elif data_type == "video":
        train_dataset = VideoDataset(data_dir, target_size, frames_per_video=16)
    else:
        raise ValueError(f"Invalid data type: {data_type}")
    
    # Create main samples directory if it doesn't exist - save generated images here
    noisy_images_dir = data_dir / f"noisy_{data_type}s"
    noisy_images_dir.mkdir(exist_ok=True, parents=True)

    # Save some noisy images before training starts
    with torch.no_grad():
        # Get random images from the dataset
        inputs, _ = next(iter(DataLoader(train_dataset, batch_size=num_images_to_save)))
        
        # Save original images first
        images = convert_neg_one_to_one_to_np_8bit(inputs)
        for i in range(num_images_to_save):
            
            if data_type == "image":
                img = Image.fromarray(images[i])
                img.save(noisy_images_dir / ( f"original_img{i+1}.png"))
            elif data_type == "video":
                # Create a list of PIL images from the video frames
                frames = [Image.fromarray(frame) for frame in images[i]]
                # Save as animated GIF
                frames[0].save(
                    noisy_images_dir / f"original_video{i+1}.gif",
                    save_all=True,
                    append_images=frames[1:],
                    duration=100, # 100ms per frame
                    loop=0
                )
        
        print(f"Saved original {data_type}s")
        
        # Create noise at different timesteps - need to pick timesteps below NUM_TIMESTEPS
        for t in timesteps:
            timesteps = torch.ones(num_images_to_save, device=inputs.device).long() * t
            noisy_images, _ = add_noise(inputs, timesteps, noise_scheduler)
            
            # Convert from [-1,1] back to [0,1] range
            noisy_images = convert_neg_one_to_one_to_np_8bit(noisy_images)
            # Convert to PIL images and save
            for i in range(num_images_to_save):
                if data_type == "image":
                    img = Image.fromarray(noisy_images[i])
                    img.save(noisy_images_dir / (f"noisy_t{t}_img{i+1}.png"))
                elif data_type == "video":
                    # Create a list of PIL images from the video frames
                    frames = [Image.fromarray(frame) for frame in noisy_images[i]]
                    # Save as animated GIF
                    frames[0].save(
                        noisy_images_dir / f"noisy_t{t}_video{i+1}.gif",
                        save_all=True,
                        append_images=frames[1:],
                        duration=100, # 100ms per frame
                        loop=0
                    )
            print(f"Saved noisy images at timestep {t}")

def save_sample(sample: torch.Tensor, save_path: Path, config: dict):

    if config['data_type'] == "image":
        sample.save(save_path)
    elif config['data_type'] == "video":
        save_video(sample, save_path)
    else:
        raise ValueError(f"Invalid data type: {config['data_type']}")

def save_video(video: np.ndarray, path: Path):
    '''
    Save a video to a file

    Args:
        video (np.ndarray): uint8 array of shape (T, H, W, 3)
        path (Path): the path to save the video to
    '''
    frames = [Image.fromarray(frame) for frame in video]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=60, loop=0)

def get_latest_model_dir(dataset_name):
    """Find the most recent model directory for the given dataset."""
    model_base_dir = Path(f"./models/{dataset_name}")
    if not model_base_dir.exists():
        raise ValueError(f"No models found for dataset {dataset_name}")
    
    # List all model directories and sort by creation time
    model_dirs = list(model_base_dir.glob("UNet2DModel*"))
    if not model_dirs:
        raise ValueError(f"No model directories found in {model_base_dir}")
    
    return max(model_dirs, key=lambda x: x.stat().st_mtime)
            
def plot_training_metrics(
        epoch_losses: List[float], 
        batch_losses: List[float], 
        timesteps_used: List[int], 
        learning_rates: List[float], 
        NUM_EPOCHS: int, 
        results_dir: Path):
    
    # After training, create and save plots
    plt.figure(figsize=(15, 5))

    # Plot 1: Epoch Losses
    plt.subplot(1, 4, 1)
    plt.semilogy(range(1, NUM_EPOCHS + 1), epoch_losses)
    plt.title('Average Loss per Epoch (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.grid(True)

    # Plot 2: Batch Losses
    plt.subplot(1, 4, 2)
    plt.semilogy(batch_losses)
    plt.title('Loss per Batch (Log Scale)') 
    plt.xlabel('Batch')
    plt.ylabel('Loss (log)')
    plt.grid(True)

    # Plot 3: Timesteps Distribution
    plt.subplot(1, 4, 3)
    plt.hist(timesteps_used, bins=50)
    plt.title('Timesteps Distribution')
    plt.xlabel('Timestep')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Plot 4: Learning Rate Schedule
    plt.subplot(1, 4, 4)
    plt.plot(learning_rates)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(results_dir / 'training_metrics.png')
    plt.close()

    print("Training metrics plot saved as: training_metrics.png")