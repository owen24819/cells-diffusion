# Standard library imports
import datetime
from pathlib import Path
import shutil
# Third-party imports
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, UNet3DConditionModel, AutoencoderKL
from tqdm import tqdm
import wandb
import numpy as np
import PIL
# Local imports
from data_utils import ImageDataset, VideoDataset
from utils import add_noise, lr_lambda, plot_training_metrics, save_noisy_images, save_video, convert_neg_one_to_one_to_np_8bit

# Training hyperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 2e-4  
NUM_TIMESTEPS = 1000
DATASET = "moma"
DATA_TYPE = "video"  # video or "image"
FRAMES_PER_VIDEO = 12  # Only used if DATA_TYPE == "video"
LATENT_CHANNELS = 4  # Number of channels in latent space
SAVE_NOISE_IMAGES = False
TARGET_SIZE = (256, 32)
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
data_dir = Path(f"./data/{DATASET}")
MODEL_NAME = f"{DATASET}_{DATA_TYPE}_UNet2DModel_BATCH_SIZE_{BATCH_SIZE}_LEARNING_RATE_{LEARNING_RATE}_EPOCHS_{NUM_EPOCHS}_NUM_TIMESTEPS_{NUM_TIMESTEPS}_{TIMESTAMP}"
model_dir = Path(f"./models/{DATASET}/{DATA_TYPE}/{MODEL_NAME}")

data_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

if DATA_TYPE == "image":
    train_dataset = ImageDataset(data_dir, TARGET_SIZE)
elif DATA_TYPE == "video":   
    train_dataset = VideoDataset(data_dir, TARGET_SIZE, FRAMES_PER_VIDEO)
else:
    raise ValueError(f"Invalid data type: {DATA_TYPE}")

train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True  # Optimizes data transfer to GPU
    )

# Initialize noise scheduler
noise_scheduler = DDPMScheduler(NUM_TIMESTEPS, beta_schedule="linear")

if SAVE_NOISE_IMAGES:
    save_noisy_images(data_dir, DATA_TYPE, TARGET_SIZE, noise_scheduler, num_images_to_save=3, timesteps=[1, 10, 100, 500])

# Initialize VAE - use same 2D VAE for both image and video
vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    subfolder="vae",
    in_channels=3,
    out_channels=3,
    torch_dtype=torch.float32,
    ignore_mismatched_sizes=True,
    low_cpu_mem_usage=False
).to(device)

# Get the VAE scale factor from the model config instead of hardcoding it
VAE_SCALE_FACTOR = 2 ** (len(vae.config.block_out_channels) - 1)

# Adjust UNet to work with latent space dimensions
if DATA_TYPE == "image":
    model = UNet2DModel(
        sample_size=(train_dataset.target_size[0] // VAE_SCALE_FACTOR, 
                    train_dataset.target_size[1] // VAE_SCALE_FACTOR),  # Reduced size in latent space
        in_channels=LATENT_CHANNELS,  # VAE latent channels
        out_channels=LATENT_CHANNELS,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)
else:  # video
    model = UNet3DConditionModel(
        sample_size=(train_dataset.target_size[0] // VAE_SCALE_FACTOR, 
                    train_dataset.target_size[1] // VAE_SCALE_FACTOR),
        in_channels=LATENT_CHANNELS,
        out_channels=LATENT_CHANNELS,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D"),
    ).to(device)

# start a new experiment
wandb.init(project=f"{DATASET}-diffusion-model-{DATA_TYPE}", name=MODEL_NAME)
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
            # Shape is [B, C, H, W] where C = 3 to be compatible with the vae
            images = images.to(device)
            
            # Encode images to latent space
            with torch.no_grad():
                if DATA_TYPE == "image":
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                else:  # video
                    # Reshape video to (batch * frames, channels, height, width)
                    b, f, c, h, w = images.shape
                    images_2d = images.reshape(-1, c, h, w)
                    # Encode each frame
                    latents = vae.encode(images_2d).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    # Reshape back to video format (batch, channels, frames, height, width)
                    _, c_latent, h_latent, w_latent = latents.shape
                    latents = latents.reshape(b, f, c_latent, h_latent, w_latent).transpose(1, 2)

            # Add noise to latents
            timesteps = torch.randint(0, NUM_TIMESTEPS, (BATCH_SIZE,), device=device).long()
            noisy_latents, noise = add_noise(latents, timesteps, noise_scheduler)

            # Predict noise
            if DATA_TYPE == "image":
                noise_pred = model(noisy_latents, timesteps).sample
            else:  # video
                encoder_hidden_states = torch.zeros(BATCH_SIZE, FRAMES_PER_VIDEO, 1024).to(device)
                noise_pred = model(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

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
    save_images_every_epoch = max(NUM_EPOCHS // 10, 1)
    if (epoch + 1) % save_images_every_epoch == 0:  # Check if current epoch is divisible by 10
        pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
        pipeline.to(device)
        
        if DATA_TYPE == "image":
            latents = pipeline(
                num_inference_steps=100, 
                batch_size=5,
                output_type="tensor",
            ).images
            # pipeline outputs [0,1] range but vae decoder expects [-1,1] range
            latents = 2 * latents - 1
            # Convert numpy array to tensor if needed
            # Shape is [B, H, W, C] because it is np array - need to convert to [B, C, H, W]
            if isinstance(latents, np.ndarray):
                latents = torch.from_numpy(latents).to(device)
                latents = latents.permute(0, 3, 1, 2)
            # Decode latents to images
            with torch.no_grad():
                images = vae.decode(latents / vae.config.scaling_factor).sample
                images = images.clip(-1, 1)
                # Convert from [B, C, H, W] to [B, H, W, C]
                images = images.permute(0, 2, 3, 1)
                images = convert_neg_one_to_one_to_np_8bit(images)
                images = [PIL.Image.fromarray(image) for image in images]
        else:  # video
            latent = torch.randn((BATCH_SIZE, LATENT_CHANNELS, FRAMES_PER_VIDEO, 
                                TARGET_SIZE[0] // VAE_SCALE_FACTOR, 
                                TARGET_SIZE[1] // VAE_SCALE_FACTOR)).to(device)
            encoder_hidden_states = torch.zeros(BATCH_SIZE, FRAMES_PER_VIDEO, 1024).to(device)
            model.eval()
            with torch.no_grad():
                # Create inference scheduler with fewer steps
                inference_scheduler = DDPMScheduler.from_config(noise_scheduler.config)
                inference_scheduler.set_timesteps(100)  # Set to 100 steps for inference
                
                for t in tqdm(inference_scheduler.timesteps, desc="Generating video"):
                    latent_input = inference_scheduler.scale_model_input(latent, t)
                    noise_pred = model(latent_input, t, encoder_hidden_states=encoder_hidden_states).sample
                    latent = inference_scheduler.step(noise_pred, t, latent).prev_sample
            
            # Decode latents to images after generation
            with torch.no_grad():
                b, c, f, h, w = latent.shape
                latent_2d = latent.transpose(1, 2).reshape(-1, c, h, w)
                images_2d = vae.decode(latent_2d / vae.config.scaling_factor).sample
                # Reshape back to video format
                _, c_out, h_out, w_out = images_2d.shape
                images = images_2d.reshape(b, f, c_out, h_out, w_out).permute(0, 1, 3, 4, 2)
        
        # Create a subdirectory for the current epoch
        epoch_dir = model_dir / f"epoch_{epoch+1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each sample
        if DATA_TYPE == "image":
            for i, sample in enumerate(images):
                image_path = epoch_dir / f"sample_{i+1}.png"
                sample.save(image_path)
        else:  # vid
            for i in range(len(images)):
                # Save as animated GIF or video file
                video_path = epoch_dir / f"sample_{i+1}.gif"
                video = convert_neg_one_to_one_to_np_8bit(images[i])
                save_video(video, video_path)
            
            print(f"Saved sample {i+1} to: {str(epoch_dir.absolute())}")

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