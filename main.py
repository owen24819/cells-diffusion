# Standard library imports
from pathlib import Path
import shutil
# Third-party imports
import torch
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from tqdm import tqdm
import wandb
# Local imports
from data_utils import create_dataset
from utils import add_noise, plot_training_metrics, save_noisy_images, save_sample
from config import get_config
from model import create_models, get_latent_from_images, get_predicted_noise, generate_samples_from_noise, get_lr_scheduler
from argparse_utils import parse_args

# Get command line arguments
args = parse_args()
DATASET = args.dataset
DATA_TYPE = args.data_type

# Get base config and update with command line arguments
config = get_config(DATA_TYPE, DATASET)
# Override config values with any non-None command line arguments
config.update({k: v for k, v in vars(args).items() if v is not None})

model_dir = Path(f"./models/{DATASET}/{DATA_TYPE}/{config['model_name']}")
data_dir = Path(f"./data/{DATASET}")

data_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

train_dataset = create_dataset(data_dir, config)

train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True  # Optimizes data transfer to GPU
    )

# Initialize noise scheduler
noise_scheduler = DDPMScheduler(config['num_timesteps'], beta_schedule="linear")

# Save noisy images to check quality of noise scheduler
if config['save_noise_images']:
    save_noisy_images(data_dir, DATA_TYPE, config['target_size'], noise_scheduler, num_images_to_save=3, timesteps=[1, 10, 100, 500])

# Initialize model
model, vae = create_models(config)

# start a new experiment
wandb.init(project=f"{DATASET}-diffusion-model-{DATA_TYPE}", name=config['model_name'])
# capture all config parameters
wandb.config.update(config)
# track gradients
wandb.watch(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
lr_scheduler = get_lr_scheduler(train_loader, optimizer, config)

# Before the training loop, initialize lists to store metrics
epoch_losses = []
batch_losses = []
timesteps_used = []
learning_rates = []

# Training loop
for epoch in range(config['num_epochs']):
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)
    batch_losses_epoch = []  # Track losses within this epoch
    current_lr = optimizer.param_groups[0]['lr']
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}") as t:
        for batch_idx, (images, _) in enumerate(t):
            # Shape is [B, C, H, W] where C = 3 to be compatible with the vae
            images = images.to(config['device'])
            
            # Encode images to latent space
            with torch.no_grad():
                latents = get_latent_from_images(images, vae, config)

            # Add noise to latents
            timesteps = torch.randint(0, config['num_timesteps'], (config['batch_size'],), device=config['device']).long()
            noisy_latents, noise = add_noise(latents, timesteps, noise_scheduler)

            # Predict noise
            noise_pred = get_predicted_noise(model, noisy_latents, timesteps, config)

            # Loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            epoch_loss += loss.item()

            # Store metrics
            batch_losses_epoch.append(loss.item())
            timesteps_used.extend(timesteps.cpu().numpy())
            
            # Backpropagate and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # Update progress bar with loss and learning rate
            t.set_postfix(
                loss=f'{loss.item():.4f}', 
                avg_loss=f'{(epoch_loss / (batch_idx + 1)):.4f}',
                lr=f'{current_lr:.6f}'
            )

            # Log metrics to wandb
            wandb.log({"loss": loss.item()})

            # Save trained model
            model.save_pretrained(model_dir)
    
    avg_loss = epoch_loss / num_batches
    epoch_losses.append(avg_loss)
    batch_losses.extend(batch_losses_epoch)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    # Generate new images every 10 epochs
    save_images_every_epoch = max(config['num_epochs'] // 10, 1)
    if (epoch + 1) % save_images_every_epoch == 0:  # Check if current epoch is divisible by 10
        
        with torch.no_grad():
            samples = generate_samples_from_noise(model, vae, noise_scheduler, config)

        # Create a subdirectory for the current epoch
        epoch_dir = model_dir / f"epoch_{epoch+1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each sample
        for i, sample in enumerate(samples):
            save_path = epoch_dir / f"sample_{i+1}.png"
            save_sample(sample, save_path, config)
        print(f"Saved samples to: {str(epoch_dir.absolute())}")
            
    # Add learning rate to the metrics plot
    learning_rates.append(current_lr)

plot_training_metrics(epoch_losses, batch_losses, timesteps_used, learning_rates, config['num_epochs'], model_dir)

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