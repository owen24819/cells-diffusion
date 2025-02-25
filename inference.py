# Standard library imports
from pathlib import Path
import datetime

# Third-party imports
import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from tqdm import tqdm
import wandb
from safetensors import safe_open

# Local imports
from utils import get_latest_model_dir

def generate_images(num_images, model_dir, output_dir, num_inference_steps=100, num_timesteps=1000):
    """Generate images using the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize noise scheduler using config if available
    noise_scheduler = DDPMScheduler(num_timesteps, beta_schedule="linear")
    
    # Load the pipeline with the model files
    config_path = model_dir / "config.json"
    model_path = model_dir / "diffusion_pytorch_model.safetensors"
    
    if not config_path.exists() or not model_path.exists():
        raise ValueError(f"Missing required model files in {model_dir}")
        
    # Create a new UNet model with the config
    unet = UNet2DModel.from_config(UNet2DModel.load_config(str(config_path)))
    
    # Create the pipeline
    pipeline = DDPMPipeline(
        unet=unet,
        scheduler=noise_scheduler
    )
    
    # Load the safetensors weights
    with safe_open(str(model_path), framework="pt", device="cpu") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
        pipeline.unet.load_state_dict(state_dict)
    
    pipeline.to(device)
    
    # Create timestamp for this generation run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    generation_dir = output_dir / f"generation_{timestamp}"
    generation_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_images} images...")
    
    # Generate images in batches of 5 to avoid memory issues
    batch_size = 5
    num_batches = (num_images + batch_size - 1) // batch_size
    
    generated_count = 0
    for batch in tqdm(range(num_batches)):
        current_batch_size = min(batch_size, num_images - generated_count)
        samples = pipeline(
            num_inference_steps=num_inference_steps,
            batch_size=current_batch_size
        ).images
        
        # Save the generated images
        for idx, img in enumerate(samples):
            image_path = generation_dir / f"generated_{num_inference_steps}_inference_steps_{generated_count + idx + 1}.png"
            img.save(image_path)
        
        generated_count += len(samples)
    
    print(f"Generated images saved to: {generation_dir}")
    return generation_dir

if __name__ == "__main__":
    # Load the latest run from W&B
    api = wandb.Api()
    runs = api.runs("cells-diffusion-model")  # Replace with your project name
    latest_run = runs[0]  # Assumes runs are sorted by date
    
    # Get configuration from the latest run
    config = latest_run.config
    DATASET = config['dataset']
    NUM_TIMESTEPS = config['timesteps']
    NUM_IMAGES = 20    # Number of images to generate
    INFERENCE_STEPS = 100  # Number of denoising steps
    
    print(f"Loaded configuration from W&B run: {latest_run.name}")
    print(f"Dataset: {DATASET}")
    print(f"Original timesteps: {config.get('timesteps')}")
    print(f"Original batch size: {config.get('batch_size')}")
    
    # Setup directories
    outputs_dir = Path("./outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Get the latest model
    try:
        model_dir = get_latest_model_dir(DATASET)
        print(f"Using model from: {model_dir}")
        
        # Generate images
        output_dir = generate_images(
            num_images=NUM_IMAGES,
            model_dir=model_dir,
            output_dir=outputs_dir,
            num_inference_steps=INFERENCE_STEPS,
            num_timesteps=NUM_TIMESTEPS,
        )
        
    except ValueError as e:
        print(f"Error: {e}") 