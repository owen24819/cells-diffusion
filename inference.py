# Standard library imports
from pathlib import Path
import datetime

# Third-party imports
import torch
from diffusers import DDPMScheduler, UNet2DModel, UNet3DConditionModel
from tqdm import tqdm
import wandb
from safetensors import safe_open

# Local imports
from utils import save_sample
from model import generate_samples_from_noise, create_models
from argparse_utils import parse_args
from config import get_config

def generate_images(num_images, output_dir, config, model_download_dir, num_inference_steps=100, device='cuda'):
    """Generate images using the trained model."""

    # Initialize model
    model, vae = create_models(config)
    
    # Load the pipeline with the model files
    config_path = model_download_dir / "config.json"
    model_path = model_download_dir / "diffusion_pytorch_model.safetensors"

    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler(config['num_timesteps'], beta_schedule="linear")
        
    # Create a new UNet model with the config
    if config['data_type'] == "image":
        model = UNet2DModel.from_config(UNet2DModel.load_config(config_path))
    elif config['data_type'] == "video":
        model = UNet3DConditionModel.from_config(UNet3DConditionModel.load_config(config_path))

    model.to(device)
    
    # Load the safetensors weights
    with safe_open(str(model_path), framework="pt", device=device) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
        model.load_state_dict(state_dict)
    
    # Create timestamp for this generation run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    generation_dir = output_dir / config['dataset'] / config['data_type'] / config['model_name'] / f"generation_{timestamp}"
    generation_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_images} images...")
    
    # Generate images in batches of 5 to avoid memory issues
    batch_size = 5
    num_batches = (num_images + batch_size - 1) // batch_size
    
    generated_count = 0
    for _ in tqdm(range(num_batches)):
        current_batch_size = min(batch_size, num_images - generated_count)
        config['batch_size'] = current_batch_size
        with torch.no_grad():
            samples = generate_samples_from_noise(model, vae, noise_scheduler, config, num_inference_steps=num_inference_steps)

        # Save each sample
        for i, sample in enumerate(samples):
            save_path = generation_dir / f"{num_inference_steps}_inference_steps_{generated_count + i + 1}.png"
            save_sample(sample, save_path, config)
        
        generated_count += len(samples)
    
    print(f"Generated images saved to: {generation_dir}")
    return generation_dir

if __name__ == "__main__":

    # Get command line arguments
    args = parse_args()
    DATASET = args.dataset # Currently only moma dataset
    DATA_TYPE = args.data_type # Currently supports images or videos
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 

    NUM_IMAGES = 20    # Number of images to generate
    INFERENCE_STEPS = 100  # Number of denoising steps

    if DATA_TYPE == "image":
        run_name = "moma-diffusion-model-image/1wpfp1pw"
    elif DATA_TYPE == "video":
        run_name = "moma-diffusion-model-video/jjbp0dqy"

    # Initialize wandb and specify the run to restore from
    api = wandb.Api()
    run = api.run("owenoconnor248-boston-university/" + run_name)

    model_name = run.config['model_name']
    wandb_path = Path(f"models/{DATASET}/{DATA_TYPE}/{model_name}")

    # Create download directory
    download_dir = Path("./downloaded_models")
    download_dir.mkdir(exist_ok=True)

    # Download the files
    for filename in ["config.json", "diffusion_pytorch_model.safetensors"]:
        file_path = f"{wandb_path}/{filename}"
        if not (download_dir / file_path).exists():
            run.file(file_path).download(root=str(download_dir))

    model_download_dir = download_dir / wandb_path
    config = run.config
    
    # Setup directories
    outputs_dir = Path("./outputs")
    outputs_dir.mkdir(exist_ok=True)
            
    # Generate images
    output_dir = generate_images(
        num_images=NUM_IMAGES,
        output_dir=outputs_dir,
        config=config,
        model_download_dir=model_download_dir,
        num_inference_steps=INFERENCE_STEPS,
        device=DEVICE,
    )