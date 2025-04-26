import torch
from sklearn.metrics import r2_score

from loss_functions import vae_loss, beta_vae_loss

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, model_type, dataloader, optimizer, criterion, device, params):
    """Training loop for one epoch."""
    model.train()

    running_loss = 0
    running_kl_div = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        if (model_type=='BetaVAE'):
            loss, kl_loss = beta_vae_loss(model, data, criterion, params['beta'])
        else:
            loss, kl_loss = vae_loss(model, data, criterion)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        running_kl_div += kl_loss.item() * data.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_kl = running_kl_div / len(dataloader.dataset)
    return epoch_loss, epoch_kl

def evaluate(model, model_type, dataloader, criterion, device, params):
    """Validation loop."""
    model.eval()
    running_loss = 0
    running_kl_div = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)

            if (model_type=='BetaVAE'):
                loss, kl_loss = beta_vae_loss(model, data, criterion, params['beta'])
            else:
                loss, kl_loss = vae_loss(model, data, criterion)

            running_loss += loss.item() * data.size(0)
            running_kl_div += kl_loss.item() * data.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_kl = running_kl_div / len(dataloader.dataset)
    return epoch_loss, epoch_kl

def get_latent_variables(model, dataloader, device):
    """Retrieve latent variables from the encoder."""
    model.eval()
    all_latent_vars = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            _, mean, _ = model(data)
            all_latent_vars.append(mean.detach().cpu())
    return torch.cat(all_latent_vars)

# evaluation.py
def calculate_mse(model, dataloader, device):
    """Calculate Mean Squared Error for the VAE reconstructions."""
    model.eval()
    mse_values = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            reconstruction, _, _ = model(data)
            mse = torch.mean((reconstruction - data) ** 2).item()
            mse_values.append(mse)
    avg_mse = sum(mse_values) / len(mse_values)
    return avg_mse