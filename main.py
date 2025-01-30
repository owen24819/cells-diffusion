import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar
import os
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load MNIST dataset only if not already downloaded
data_dir = "./data"
if not os.path.exists(os.path.join(data_dir, "MNIST")):
    download = True
else:
    download = False

# Create main samples directory if it doesn't exist - save generated images here
os.makedirs("samples", exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 2 * x - 1)  # Normalize to [-1,1]
])

train_dataset = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Smaller batch size for CPU

# Define a small UNet model
model = UNet2DModel(
    sample_size=28,  # Image size
    in_channels=1,   # Grayscale images
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(32, 64, 128),  # Small model
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(device)

# Define noise scheduler with fewer steps
scheduler = DDPMScheduler(num_train_timesteps=200)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# Training loop
epochs = 3  # Reduce epochs to avoid long training times on CPU
for epoch in range(epochs):
    epoch_loss = 0.0
    num_batches = len(train_loader)
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as t:
        for batch_idx, (images, _) in enumerate(t):
            images = images.to(device)

            # Add noise
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
            noisy_images = scheduler.add_noise(images, noise, timesteps)

            # Predict noise
            noise_pred = model(noisy_images, timesteps).sample

            # Loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            epoch_loss += loss.item()

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss_so_far = epoch_loss / (batch_idx + 1)
            t.set_postfix(loss=loss.item(), avg_loss=avg_loss_so_far)  # Update progress bar with loss and avg loss
    
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    # Generate new images after each epoch
    pipeline = DDPMPipeline(unet=model, scheduler=scheduler)
    pipeline.to(device)
    samples = pipeline(num_inference_steps=30, batch_size=5).images

    # Create a subdirectory for the current epoch
    epoch_dir = os.path.join("samples", f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Save each sample as a PNG file
    for i, img in enumerate(samples):
        # Save PIL Image directly
        image_path = os.path.join(epoch_dir, f"sample_{i+1}.png")
        img.save(image_path)
        
        # Print path for easy viewing in VSCode
        print(f"Saved sample {i+1} to: {os.path.abspath(image_path)}")

# Save trained model
model.save_pretrained("./mnist_diffusion_cpu")
