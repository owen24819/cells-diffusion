import torch
from torch.utils.data import DataLoader

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List
import math
import numpy as np

from diffusers import DDPMScheduler

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
        train_dataset: torch.utils.data.Dataset, 
        noisy_images_dir: Path, 
        noise_scheduler: DDPMScheduler,
        timesteps: List[int],
        num_images_to_save: int, 
        ):
    
    # Create main samples directory if it doesn't exist - save generated images here
    noisy_images_dir.mkdir(exist_ok=True, parents=True)

    # Save some noisy images before training starts
    with torch.no_grad():
        # Get random images from the dataset
        sample_images, _ = next(iter(DataLoader(train_dataset, batch_size=num_images_to_save)))
        sample_images = sample_images
        
        # Save original images first
        for i in range(num_images_to_save):
            img = convert_neg_one_to_one_to_np_8bit(sample_images[i])
            img = Image.fromarray(img)
            img.save(noisy_images_dir / ( f"original_img{i+1}.png"))
        
        print("Saved original images")
        
        # Create noise at different timesteps - need to pick timesteps below NUM_TIMESTEPS
        for t in timesteps:
            timesteps = torch.ones(num_images_to_save, device=sample_images.device).long() * t
            noisy_images, _ = add_noise(sample_images, timesteps, noise_scheduler)
            
            # Convert to PIL images and save
            for i in range(num_images_to_save):
                # Convert from [-1,1] back to [0,1] range
                img = convert_neg_one_to_one_to_np_8bit(noisy_images[i])
                img = Image.fromarray(img)
                img.save(noisy_images_dir / (f"noisy_t{t}_img{i+1}.png"))
            print(f"Saved noisy images at timestep {t}")
            
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
    plt.semilogy(range(1, NUM_EPOCHS + 1), epoch_losses, marker='o')
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