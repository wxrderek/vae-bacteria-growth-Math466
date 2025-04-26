import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import os
import argparse

from data_loading import load_simulated_data
from training import train_epoch, evaluate, count_parameters
from utils import setup_logging, create_output_dir
from plotting import plot_loss, plot_kl, plot_reconstructions, plot_sample_trajectories

# import models
from models.shallow_vae import VAE
from models.deep_cnn_vae import DeepCNNVAE
from models.beta_vae import BetaVAE
from models.info_vae import InfoVAE


def main(model_type='VAE'):
    # setup logging
    logger = setup_logging()
    logger.info(f"Model type received: {model_type}")

    # common hyperparameters
    latent_dim = 12
    params = {
        'beta': 1e-4, # beta VAE
        'alpha': 0.5, # info VAE
        'lambda_': 0.3 # info VAE
    }

    # hyperparameters for linear
    input_dim = 600
    hidden_dim = 32

    # hyperparameters for CNN
    batch_size = 32
    latent_channel = 16
    lr = 1e-3            
    min_lr = 4e-6 
    epochs = 20
    gamma = 0.98
    weight_decay = 1e-5

    # set data directory based on distribution type
    data_directory = 'simulated_data/'
    logger.info(f"Using data directory: {data_directory}")

    # Create output directory with distribution information
    output_dir = create_output_dir(
        base_output_dir='train_vae_output',
        model_type=model_type,
        latent_dim=latent_dim,
        latent_channel=latent_channel,
        lr=lr, 
        params=params
    )

    # Additionally, set up a subdirectory for models
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # load and process simulated data
    logger.info(f"Loading simulated data from {data_directory}")
    all_data = load_simulated_data(data_directory, logger)
    np.random.shuffle(all_data)
    logger.info(f"Combined data shape after shuffling: {all_data.shape}")

    # prepare data
    data = all_data  
    seq_length = data.shape[1]
    data = torch.tensor(data).float().unsqueeze(1)

    # determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"CUDA is available. Device count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        logger.warning("CUDA is not available. Check your CUDA installation and NVIDIA drivers.")

    # split the data
    train_data, test_data, _, _ = train_test_split(
        data, range(data.shape[0]), test_size=0.1, random_state=42
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

    # initialize model based on the selected architecture
    if model_type == 'VAE':
        model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    elif model_type == 'DeepCNNVAE':
        model = DeepCNNVAE(latent_dim=latent_dim, latent_channel=latent_channel, seq_length=seq_length)
    elif model_type == 'BetaVAE':
        model = BetaVAE(latent_dim=latent_dim, latent_channel=latent_channel, seq_length=seq_length)
    elif model_type == 'InfoVAE':
        model = InfoVAE(latent_dim=latent_dim, latent_channel=latent_channel, seq_length=seq_length)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"The model {model_type} is currently being trained")
    model = model.to(device)

    num_params = count_parameters(model)
    logger.info(f'The model has {num_params:,} parameters')

    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # learning rate schedulers
    warmup_epochs = 10
    scheduler1 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0)
    scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # training loop variables
    train_loss_values = []
    test_loss_values = []
    train_kl_loss_values = []
    test_kl_loss_values = []
    best_test_loss = np.inf
    best_state_dict = None
    epochs_no_improve = 0
    patience = 30

    logger.info("Starting training loop...")
    for epoch in range(epochs):
        # train for one epoch
        train_loss, train_kl_loss = train_epoch(model, model_type, train_loader, optimizer, criterion, device, params)
        # evaluate on test set
        test_loss, test_kl_loss = evaluate(model, model_type, test_loader, criterion, device, params)

        # record losses
        train_loss_values.append(train_loss)
        test_loss_values.append(test_loss)
        train_kl_loss_values.append(train_kl_loss)
        test_kl_loss_values.append(test_kl_loss)

        # update learning rate
        if epoch < warmup_epochs:
            scheduler1.step()
        else:
            scheduler2.step()

        # clamp minimum learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < min_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = min_lr
            current_lr = min_lr

        # logging (less frequent logging after first 10 epochs)
        interval = 2 if epoch < 10 else 40
        if (epoch + 1) % interval == 0:
            logger.info(f'Epoch: {epoch + 1} | '
                        f'Train Loss: {train_loss:.7f}, '
                        f'Test Loss: {test_loss:.7f}, '
                        f'Lr: {current_lr:.8f}')

        # early stopping check
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0
            # update best_state_dict
            best_state_dict = model.state_dict()
        else:
            epochs_no_improve += 1

        # trigger early stopping
        if epochs_no_improve >= patience:
            logger.info('Early stopping triggered!')
            break

    logger.info("Training completed.")

    # save  best model
    if best_state_dict is not None:
        best_model_path = os.path.join(models_dir, 'best_model.pt')
        torch.save(best_state_dict, best_model_path)
        logger.info(f"Best model saved to {best_model_path}")
    else:
        logger.warning("No improvement during training. Best model not saved.")

    # plot loss curves
    plot_loss(train_loss_values, test_loss_values, output_dir)
    plot_kl(train_kl_loss_values, test_kl_loss_values, output_dir)

    # retrieve a subset of data for reconstruction plots
    percentage = 0.2
    num_train_samples = int(len(train_data) * percentage)
    num_test_samples = int(len(test_data) * percentage)

    subset_train_data = train_data[:num_train_samples].to(device)
    subset_test_data = test_data[:num_test_samples].to(device)

    # plot reconstructions
    plot_reconstructions(model, model_type, subset_train_data, subset_test_data, output_dir, device)

    # plot sample trajectories
    plot_sample_trajectories(model, model_type, subset_train_data, subset_test_data, output_dir, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE Models')
    parser.add_argument('--model', type=str, default='VAE',
                        choices=['VAE', 'DeepCNNVAE', 'BetaVAE', 'InfoVAE'],
                        help='Specify which VAE architecture to use')
    args = parser.parse_args()

    main(model_type=args.model)