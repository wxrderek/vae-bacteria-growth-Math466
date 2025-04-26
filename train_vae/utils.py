import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_file='vae_training.log'):
    """Setup logging configuration."""
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    return logger

def create_output_dir(base_output_dir, model_type, latent_dim, latent_channel, lr, params):
    """
    Create a unique output directory based on model configuration and timestamp.
    
    Parameters:
    - base_output_dir (str): Base directory for outputs.
    - model_type (str): Type/name of the model architecture.
    - latent_dim (int): Dimension of the latent space.
    - latent_channel (int): Number of latent channels.
    - lr (float): Learning rate.
    - distribution_type (str): Type of distribution used for simulation.
    
    Returns:
    - output_dir (str): Path to the created output directory.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if (model_type=='VAE'):
        dir_name = f"{model_type}_LD{latent_dim}_LR{lr}_TS{timestamp}"
    elif (model_type=='BetaVAE'):
        dir_name = f"{model_type}_BETA{params['beta']}_LD{latent_dim}_LC{latent_channel}_LR{lr}_TS{timestamp}"
    elif (model_type=='InfoVAE'):
        dir_name = f"{model_type}_ALPHA{params['alpha']}_LAMBDA_{params['lambda_']}_LD{latent_dim}_LC{latent_channel}_LR{lr}_TS{timestamp}"
    else: dir_name = f"{model_type}_LD{latent_dim}_LC{latent_channel}_LR{lr}_TS{timestamp}"
    
    # Create the full path
    output_dir = Path(base_output_dir) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return str(output_dir)