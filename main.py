# Standard library imports
import datetime
from pathlib import Path
import shutil
# Third-party imports
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from tqdm import tqdm
import wandb

# Local imports
from data_utils import load_dataset
from utils import add_noise, lr_lambda, plot_training_metrics, save_noisy_images

# Training hyperparameters
BATCH_SIZE = 4  
NUM_EPOCHS = 100
LEARNING_RATE = 2e-4  
NUM_TIMESTEPS = 1000
DATASET = "cells"
SAVE_NOISE_IMAGES = True
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = Path(f"./data/{DATASET}")
MODEL_NAME = f"UNet2DModel_BATCH_SIZE_{BATCH_SIZE}_LEARNING_RATE_{LEARNING_RATE}_EPOCHS_{NUM_EPOCHS}_NUM_TIMESTEPS_{NUM_TIMESTEPS}_{TIMESTAMP}"
model_dir = Path(f"./models/{DATASET}/{MODEL_NAME}")

data_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

train_dataset = load_dataset(data_dir)

train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True  # Optimizes data transfer to GPU
    )

# Initialize noise scheduler
noise_scheduler = DDPMScheduler(NUM_TIMESTEPS, beta_schedule="linear")

if SAVE_NOISE_IMAGES:
    # Create a dataset for noise visualization
    noise_vis_dataset = load_dataset(data_dir)
    save_noisy_images(
        train_dataset = noise_vis_dataset, 
        noisy_images_dir = data_dir / "noisy_images", 
        noise_scheduler = noise_scheduler, 
        num_images_to_save = 3, 
        timesteps = [1, 10, 100, 500]
    )

# Initialize larger model with better architecture
model = UNet2DModel(
    sample_size=train_dataset.target_size,  # Image size
    in_channels=1,   # Grayscale images
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 256),  # Made channels increase consistently
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),  # Removed attention for now
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),  # Removed attention for now
).to(device)

# start a new experiment
wandb.init(project="cells-diffusion-model", name=MODEL_NAME)
# capture a dictionary of hyperparameters with config
wandb.config.update({
    "learning_rate": LEARNING_RATE, 
    "epochs": NUM_EPOCHS, 
    "batch_size": BATCH_SIZE, 
    "timesteps": NUM_TIMESTEPS, 
    "dataset": DATASET
    })

# track gradients
wandb.watch(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Model parameters and optimization
total_steps = NUM_EPOCHS * len(train_loader)
warmup_steps = int(0.05 * total_steps)  # Reduced to 5% warmup
lr_scheduler = lambda step: lr_lambda(step, warmup_steps, total_steps)
scheduler = LambdaLR(optimizer, lr_scheduler)

# Before the training loop, initialize lists to store metrics
epoch_losses = []
batch_losses = []
timesteps_used = []
learning_rates = []

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)
    batch_losses_epoch = []  # Track losses within this epoch
    current_lr = optimizer.param_groups[0]['lr']
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}") as t:
        for batch_idx, (images, _) in enumerate(t):
            images = images.to(device)  # Ensure data is on GPU

            # Add noise to images
            timesteps = torch.randint(0, NUM_TIMESTEPS, (images.shape[0],), device=device).long()
            noisy_images, noise = add_noise(images, timesteps, noise_scheduler)

            # Predict noise
            noise_pred = model(noisy_images, timesteps).sample

            # Loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            epoch_loss += loss.item()

            # Store metrics
            batch_losses_epoch.append(loss.item())
            timesteps_used.extend(timesteps.cpu().numpy())
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # Update progress bar with loss and learning rate
            t.set_postfix(
                loss=f'{loss.item():.4f}', 
                avg_loss=f'{(epoch_loss / (batch_idx + 1)):.4f}',
                lr=f'{current_lr:.6f}'
            )

            wandb.log({"loss": loss.item()})

            # Save trained model
            model.save_pretrained(model_dir)
    
    avg_loss = epoch_loss / num_batches
    epoch_losses.append(avg_loss)
    batch_losses.extend(batch_losses_epoch)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    # Generate new images every 10 epochs
    save_images_every_epoch = max(NUM_EPOCHS // 5, NUM_EPOCHS)
    if (epoch + 1) % save_images_every_epoch == 0:  # Check if current epoch is divisible by 10
        pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
        pipeline.to(device)
        samples = pipeline(num_inference_steps=100, batch_size=5).images

        # Create a subdirectory for the current epoch
        epoch_dir = model_dir / f"epoch_{epoch+1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each sample as a PNG file
        for i, img in enumerate(samples):
            # Save PIL Image directly
            image_path = epoch_dir / f"sample_{i+1}.png"
            img.save(image_path)
            
            # Print path for easy viewing in VSCode
            print(f"Saved sample {i+1} to: {image_path.absolute()}")

    # Add learning rate to the metrics plot
    learning_rates.append(current_lr)

plot_training_metrics(epoch_losses, batch_losses, timesteps_used, learning_rates, NUM_EPOCHS, model_dir)

##TODO This is repetitive, we should save in one spot; will move to wand once figured out
# Save trained model
model.save_pretrained(model_dir)

wandb_dir = Path(wandb.run.dir)  # Get the W&B run directory

# Copy files to the W&B directory manually since wandb.save isn't working due symlink / window permisions issues
shutil.copytree(model_dir, wandb_dir, dirs_exist_ok=True)

# Manually log the copied files as an artifact
artifact = wandb.Artifact("diffusers_model", type="model")
artifact.add_dir(model_dir)
wandb.log_artifact(artifact)