import torch
from diffusers import AutoencoderKL
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from data_utils import VideoDataset
from PIL import Image

def process_videos(data_dir: Path, output_dir: Path, target_size=(256, 32), frames_per_video=16):
    # Create output directory if it doesn't exist
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the pre-trained VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        subfolder="vae",
        in_channels=3,
        out_channels=3,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False
    ).to(device)
    vae.eval()

    # Create dataset
    dataset = VideoDataset(data_dir, target_size=target_size, frames_per_video=frames_per_video)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(dataloader)):
            # images shape: [1, frames_per_video, 3, H, W]
            images = images.squeeze(0).to(device)  # Remove batch dimension
            
            # Process each frame in the sequence
            decoded_frames = []
            for frame in images:
                # Encode and decode with VAE
                latents = vae.encode(frame.unsqueeze(0)).latent_dist.sample()
                decoded = vae.decode(latents).sample
                
                # Convert back to image
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                decoded = (decoded[0].cpu().numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
                decoded_frames.append(decoded)
            
            # Convert frames to PIL Images and save as GIF
            pil_frames = [Image.fromarray(frame) for frame in decoded_frames]
            output_path = output_dir / f"processed_video_{batch_idx:04d}.gif"
            
            # Save as GIF with 33ms delay between frames (approximately 30 fps)
            pil_frames[0].save(
                str(output_path),
                save_all=True,
                append_images=pil_frames[1:],
                duration=100,  # milliseconds between frames
                loop=0       # 0 means loop forever
            )

if __name__ == "__main__":

    DATASET = "moma"
    data_dir = Path(f"./data/{DATASET}")
    output_dir = Path(f"./data/{DATASET}/processed_videos")
    
    process_videos(
        data_dir=data_dir,
        output_dir=output_dir,
        target_size=(256, 32),  # Adjust based on your needs
        frames_per_video=16      # Adjust based on your needs
    )