import torch
import datetime

# Configuration dictionaries for image and video settings
IMAGE_CONFIG = {
    "data_type": "image",
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 2e-4,
    "num_timesteps": 1000,
    "target_size": (256, 32),
    "latent_channels": 4,
    "save_noise_images": False,
}

VIDEO_CONFIG = {
    "data_type": "video",
    "batch_size": 4,
    "num_epochs": 10,
    "learning_rate": 2e-4,
    "num_timesteps": 1000,
    "target_size": (256, 32),
    "latent_channels": 4,
    "save_noise_images": False,
    "frames_per_video": 12,
}


def get_config(data_type, dataset):
    if data_type == "image":
        config = IMAGE_CONFIG
    elif data_type == "video":
        config = VIDEO_CONFIG
    else:
        raise ValueError(f"Invalid data type: {data_type}")
    
    config['dataset'] = dataset

    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config['model_name'] = f"{dataset}_{data_type}_UNet2DModel_BATCH_SIZE_{config['batch_size']}_LEARNING_RATE_{config['learning_rate']}_EPOCHS_{config['num_epochs']}_NUM_TIMESTEPS_{config['num_timesteps']}_{TIMESTAMP}"

    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    return config