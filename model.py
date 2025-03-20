from diffusers import UNet2DModel, UNet3DConditionModel, AutoencoderKL
import torch
import PIL
from tqdm import tqdm
from utils import convert_neg_one_to_one_to_np_8bit
from diffusers import DDPMPipeline, DDPMScheduler
from torch.optim.lr_scheduler import LambdaLR

# Local imports
from utils import lr_lambda
def create_unet(config):
    """Initialize and return the UNet model based on data type."""
    # Calculate latent dimensions
    latent_height = config['target_size'][0] // config['vae_scale_factor']
    latent_width = config['target_size'][1] // config['vae_scale_factor']
    
    if config['data_type'] == "image":
        model = UNet2DModel(
            sample_size=(latent_height, latent_width),
            in_channels=config['latent_channels'],
            out_channels=config['latent_channels'],
            layers_per_block=2,
            block_out_channels=(64, 128, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        ).to(config['device'])
    else:  # video
        model = UNet3DConditionModel(
            sample_size=(latent_height, latent_width),
            in_channels=config['latent_channels'],
            out_channels=config['latent_channels'],
            layers_per_block=2,
            block_out_channels=(64, 128, 256),
            down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D"),
            up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D"),
        ).to(config['device'])
    
    return model 

def create_vae(config):
    """Initialize and return the VAE model."""
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        subfolder="vae",
        in_channels=3,
        out_channels=3,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False
    ).to(config['device'])
    
    # Calculate and add VAE scale factor to config
    config['vae_scale_factor'] = 2 ** (len(vae.config.block_out_channels) - 1)
    
    return vae

def create_models(config):
    """Initialize and return the model."""
    vae = create_vae(config)
    unet = create_unet(config)
    return unet, vae

def get_latent_from_images(images, vae, config):
    if config['data_type'] == "image":
        # Encode images to latent space
        latents = vae.encode(images).latent_dist.sample()
        # Scale latents by VAE scaling factor
        latents = latents * vae.config.scaling_factor
    elif config['data_type'] == "video":
        # Reshape video to (batch * frames, channels, height, width)
        b, f, c, h, w = images.shape
        images_2d = images.reshape(-1, c, h, w)
        # Encode each frame
        latents = vae.encode(images_2d).latent_dist.sample()
        # Scale latents by VAE scaling factor
        latents = latents * vae.config.scaling_factor
        # Reshape back to video format (batch, frames, latent channels, latent height, latent width)
        _, c_latent, h_latent, w_latent = latents.shape
        latents = latents.reshape(b, f, c_latent, h_latent, w_latent).transpose(1, 2)
    else:
        raise ValueError(f"Invalid data type: {config['data_type']}")

    return latents

def get_predicted_noise(model, latents, timesteps, config):
    if config['data_type'] == "image":
        predicted_noise = model(latents, timesteps).sample
    elif config['data_type'] == "video":
        encoder_hidden_states = torch.zeros(config['batch_size'], config['frames_per_video'], 1024).to(config['device'])
        predicted_noise = model(latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
    else:
        raise ValueError(f"Invalid data type: {config['data_type']}")
    
    return predicted_noise

def generate_samples_from_noise(model, vae, noise_scheduler, config, batch_size=None, num_inference_steps=100):

    batch_size = config['batch_size'] if batch_size is None else batch_size

    model.eval()

    if config['data_type'] == "image":
        pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
        pipeline.to(config['device'])
        pred_latents = pipeline(
            num_inference_steps=100, 
            batch_size=batch_size,
            output_type="tensor",
        ).images

        # Output is np array, convert to tensor and permute from [B, H, W, C] to [B, C, H, W]
        pred_latents = torch.from_numpy(pred_latents).to(config['device'])
        pred_latents = pred_latents.permute(0, 3, 1, 2)

        # pipeline outputs [0,1] range but vae decoder expects [-1,1] range
        pred_latents = 2 * pred_latents - 1
        
        # Decode latents to images
        samples = vae.decode(pred_latents / vae.config.scaling_factor).sample
        samples = samples.clip(-1, 1)
        # Convert from [B, C, H, W] to [B, H, W, C]
        samples = samples.permute(0, 2, 3, 1)

    elif config['data_type'] == "video":

        latent = torch.randn(
                              (batch_size,
                              config['latent_channels'], config['frames_per_video'], 
                              config['target_size'][0] // config['vae_scale_factor'], 
                              config['target_size'][1] // config['vae_scale_factor'])
                              ).to(config['device'])
        
        encoder_hidden_states = torch.zeros(batch_size, config['frames_per_video'], 1024).to(config['device'])
        # Create inference scheduler with fewer steps
        inference_scheduler = DDPMScheduler.from_config(noise_scheduler.config)
        inference_scheduler.set_timesteps(num_inference_steps)  # Set to 100 steps for inference
        
        for t in tqdm(inference_scheduler.timesteps, desc="Generating video"):
            latent_input = inference_scheduler.scale_model_input(latent, t)
            noise_pred = model(latent_input, t, encoder_hidden_states=encoder_hidden_states).sample
            latent = inference_scheduler.step(noise_pred, t, latent).prev_sample
    
        # Decode latents to samples after generation
        b, c, f, h, w = latent.shape
        latent = latent.transpose(1, 2).reshape(-1, c, h, w)
        samples = vae.decode(latent / vae.config.scaling_factor).sample
        samples = samples.clip(-1, 1)

        # Reshape back to video format
        _, c_out, h_out, w_out = samples.shape
        samples = samples.reshape(b, f, c_out, h_out, w_out).permute(0, 1, 3, 4, 2)

    else:
        raise ValueError(f"Invalid data type: {config['data_type']}")
    
    # Convert to numpy 8-bit
    samples = convert_neg_one_to_one_to_np_8bit(samples)

    return samples

def get_lr_scheduler(train_loader, optimizer, config):
    total_steps = config['num_epochs'] * len(train_loader)
    warmup_steps = int(0.05 * total_steps)  # Reduced to 5% warmup
    lr_scheduler = lambda step: lr_lambda(step, warmup_steps, total_steps)
    scheduler = LambdaLR(optimizer, lr_scheduler)

    return scheduler