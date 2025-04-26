import numpy as np
import torch

# classical VAE objective
def vae_loss(model, data, criterion):
    reconstruction, mean, logvar, z = model(data)
    recon_loss = criterion(reconstruction, data)
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    loss = recon_loss + kl_loss
    return loss, kl_loss

# beta VAE objective
def beta_vae_loss(model, data, criterion, beta):
    reconstruction, mean, logvar, z = model(data)
    recon_loss = criterion(reconstruction, data)
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    loss = recon_loss + beta * kl_loss
    return loss, kl_loss

# MMD for InfoVAE
def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd

def mmd_loss(model, data, criterion, true_samples, alpha, lambda_):
    reconstruction, mean, logvar, z = model(data)
    recon_loss = criterion(reconstruction, data)
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    mmd = compute_mmd(true_samples, z)

    loss = recon_loss + (1 - alpha) * kl_loss - (alpha + lambda_ - 1) * mmd
    return loss, kl_loss, mmd

# ladder vae objective
def ladder_loss(model, data, criterion, beta):
    reconstruction, means, logvars, zs = model(data)
    recon_loss = criterion(reconstruction, data)

    kl_loss = 0
    for mean, logvar in zip(means, logvars):
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss += kl
    
    loss = recon_loss + beta * kl_loss
    return loss, kl_loss



