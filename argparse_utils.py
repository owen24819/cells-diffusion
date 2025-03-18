import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a diffusion model')
    
    # Required arguments
    parser.add_argument('--dataset', type=str, default="moma",
                      help='Dataset to use for training')
    parser.add_argument('--data-type', type=str, default="video", choices=['video', 'image'],
                      help='Type of data to process (video or image)')
    
    # Optional training hyperparameters
    parser.add_argument('--batch-size', type=int,
                      help='Batch size for training (overrides config)')
    parser.add_argument('--learning-rate', type=float,
                      help='Learning rate (overrides config)')
    parser.add_argument('--num-epochs', type=int,
                      help='Number of training epochs (overrides config)')
    parser.add_argument('--num-timesteps', type=int,
                      help='Number of timesteps for diffusion (overrides config)')
    parser.add_argument('--target-size', type=int,
                      help='Target size for images/frames (overrides config)')
    parser.add_argument('--latent-channels', type=int,
                      help='Number of latent channels (overrides config)')
    parser.add_argument('--save-noise-images', type=bool,
                      help='Save noisy images (overrides config)')
    parser.add_argument('--frames-per-video', type=int,
                      help='Number of frames per video (overrides config)')
    parser.add_argument('--model-name', type=str,
                      help='Name of the model (overrides config)')
    
    args = parser.parse_args()
    return args