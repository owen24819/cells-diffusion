# Standard library imports
import datetime
from pathlib import Path
import shutil
# Third-party imports
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, UNet3DConditionModel
from tqdm import tqdm
import wandb

# Local imports
from data_utils import ImageDataset, VideoDataset
from utils import add_noise, lr_lambda, plot_training_metrics, save_noisy_images, save_video, convert_neg_one_to_one_to_np_8bit

# Training hyperparameters
BATCH_SIZE = 1
NUM_EPOCHS = 20
LEARNING_RATE = 2e-4  
NUM_TIMESTEPS = 1000
DATASET = "moma"
DATA_TYPE = "video"  # video or "image"
FRAMES_PER_VIDEO = 8  # Only used if DATA_TYPE == "video"
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
        pin_memory=True  # Optimizes data transfer to GPU
    )

# Initialize noise scheduler
noise_scheduler = DDPMScheduler(NUM_TIMESTEPS, beta_schedule="linear")

if SAVE_NOISE_IMAGES:
    save_noisy_images(data_dir, DATA_TYPE, TARGET_SIZE, noise_scheduler, num_images_to_save=3, timesteps=[1, 10, 100, 500])

# Initialize model based on data type
if DATA_TYPE == "image":
    model = UNet2DModel(
        sample_size=train_dataset.target_size,  # Image size
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)
else:  # video
    model = UNet3DConditionModel(
        sample_size=train_dataset.target_size,  # (height, width)
        in_channels=1,  # single channel per frame
        out_channels=1,
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
            images = images.to(device)  # Ensure data is on GPU

            # Add noise to images
            timesteps = torch.randint(0, NUM_TIMESTEPS, (BATCH_SIZE,), device=device).long()
            noisy_images, noise = add_noise(images, timesteps, noise_scheduler)

            # Predict noise
            if DATA_TYPE == "image":
                noise_pred = model(noisy_images, timesteps).sample
            else:  # video
                # For UNet3DUnconditionalModel, we need to reshape the input
                encoder_hidden_states = torch.zeros(BATCH_SIZE, FRAMES_PER_VIDEO, 1024).to(device)  # Changed from (BATCH_SIZE, 1, 1024)
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states).sample

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
    save_images_every_epoch = max(NUM_EPOCHS // 5, 1)
    if (epoch + 1) % save_images_every_epoch == 0:  # Check if current epoch is divisible by 10
        pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
        pipeline.to(device)
        
        if DATA_TYPE == "image":
            samples = pipeline(
                num_inference_steps=100, 
                batch_size=5
            ).images
        else:  # video
            # For video generation
            latent = torch.randn((BATCH_SIZE, model.config.in_channels, FRAMES_PER_VIDEO, *TARGET_SIZE)).to(device)
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
        
        # Create a subdirectory for the current epoch
        epoch_dir = model_dir / f"epoch_{epoch+1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each sample
        if DATA_TYPE == "image":
            for i, sample in enumerate(samples):
                image_path = epoch_dir / f"sample_{i+1}.png"
                sample.save(image_path)
        else:  # vid
            for i in range(len(latent)):
                # Save as animated GIF or video file
                video_path = epoch_dir / f"sample_{i+1}.gif"
                video = convert_neg_one_to_one_to_np_8bit(latent[i,0])
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