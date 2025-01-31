import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar
import os
from PIL import Image
import math
from torch.optim.lr_scheduler import LambdaLR

# Training hyperparameters
batch_size = 256  # Increased for better statistics
num_epochs = 100  # Train for longer
num_timesteps = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Model parameters and optimization
learning_rate = 2e-4  # Slightly increased
total_steps = num_epochs * len(train_loader)
warmup_steps = int(0.05 * total_steps)  # Reduced to 5% warmup

# Initialize larger model with better architecture
model = UNet2DModel(
    sample_size=28,  # Image size
    in_channels=1,   # Grayscale images
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(32, 64, 128),  # Small model
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Custom learning rate schedule with warmup and cosine decay
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps  # Linear warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

scheduler = LambdaLR(optimizer, lr_lambda)

# Before the training loop, initialize lists to store metrics
epoch_losses = []
batch_losses = []
timesteps_used = []
learning_rates = []

# Create directory for noisy images
noisy_images_dir = os.path.join("samples", "noisy_images")
os.makedirs(noisy_images_dir, exist_ok=True)

# Initialize noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_timesteps,
    beta_schedule="linear",  
)

def add_noise(images, timesteps, device):
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

# Save some noisy images before training starts
with torch.no_grad():
    # Get 3 random images from the dataset
    num_images_to_save = 3
    sample_images, _ = next(iter(DataLoader(train_dataset, batch_size=num_images_to_save)))
    sample_images = sample_images.to(device)
    
    # Save original images first
    for i in range(num_images_to_save):
        # Convert from [-1,1] back to [0,1] range
        img_array = ((sample_images[i].cpu().squeeze() + 1) / 2).numpy()
        img = Image.fromarray((img_array * 255).astype('uint8'))
        img.save(os.path.join(noisy_images_dir, f"original_img{i+1}.png"))
    
    print("Saved original images")
    
    # Create noise at different timesteps - need to pick timesteps below num_timesteps
    for t in [1, 10, 100, 500]:
        timesteps = torch.ones(num_images_to_save, device=device).long() * t
        noisy, _ = add_noise(sample_images, timesteps, device)
        
        # Convert to PIL images and save
        for i in range(num_images_to_save):
            # Convert from [-1,1] back to [0,1] range
            img_array = ((noisy[i].cpu().squeeze() + 1) / 2).numpy()
            img = Image.fromarray((img_array * 255).astype('uint8'))
            img.save(os.path.join(noisy_images_dir, f"noisy_t{t}_img{i+1}.png"))
        
        print(f"Saved noisy images at timestep {t}")

# Training loop
epochs = 3  # Reduce epochs to avoid long training times on CPU
for epoch in range(epochs):
    epoch_loss = 0.0
    num_batches = len(train_loader)
    batch_losses_epoch = []  # Track losses within this epoch
    current_lr = optimizer.param_groups[0]['lr']
    
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
    
    avg_loss = epoch_loss / num_batches
    epoch_losses.append(avg_loss)
    batch_losses.extend(batch_losses_epoch)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    # Generate new images every 10 epochs
    if (epoch + 1) % 10 == 0:  # Check if current epoch is divisible by 10
        pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
        pipeline.to(device)
        samples = pipeline(num_inference_steps=100, batch_size=5).images

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

    # Add learning rate to the metrics plot
    learning_rates.append(current_lr)

# After training, create and save plots
plt.figure(figsize=(15, 5))

# Plot 1: Epoch Losses
plt.subplot(1, 4, 1)
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
plt.title('Average Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Plot 2: Batch Losses
plt.subplot(1, 4, 2)
plt.plot(batch_losses)
plt.title('Loss per Batch')
plt.xlabel('Batch')
plt.ylabel('Loss')
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
plt.savefig('samples/training_metrics.png')
plt.close()

print("Training metrics plot saved as: training_metrics.png")

# Save trained model
model.save_pretrained("./mnist_diffusion_cpu")
