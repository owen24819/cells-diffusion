# Standard library imports
from pathlib import Path

# Third party imports
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

# Local imports
from argparse_utils import parse_args
from config import get_config
from utils import save_noisy_samples
from data_utils import download_data, create_dataset

TIMESTEPS = [1, 10, 100, 500]
NUM_IMAGES_TO_SAVE = 5

# Get command line arguments
args = parse_args()
DATASET = args.dataset # Currently only moma dataset
DATA_TYPE = args.data_type # Currently supports images or videos

# Get base config and update with command line arguments
config = get_config(DATA_TYPE, DATASET)
# Override config values with any non-None command line arguments
config.update({k: v for k, v in vars(args).items() if v is not None})

data_dir = Path(f"./data/{DATASET}")
data_dir = download_data(data_dir, DATA_TYPE)

train_dataset = create_dataset(data_dir / "train", config)

train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=NUM_IMAGES_TO_SAVE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True  # Optimizes data transfer to GPU
    )

# Initialize noise scheduler
noise_scheduler = DDPMScheduler(config['num_timesteps'], beta_schedule="linear")

save_noisy_samples(
    data_dir / f"noisy_{DATA_TYPE}s", 
    DATA_TYPE,
    train_loader, 
    noise_scheduler, 
    TIMESTEPS,
    config
    )
