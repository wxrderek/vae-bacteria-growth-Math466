import numpy as np
import torch

def vae_loss(model, data, criterion):
    reconstruction, mean, logvar = model(data)
    recon_loss = criterion(reconstruction, data)
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    loss = recon_loss + kl_loss
    return loss, kl_loss

def beta_vae_loss(model, data, criterion, beta):
    reconstruction, mean, logvar = model(data)
    recon_loss = criterion(reconstruction, data)
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl_loss
    return loss, kl_loss
