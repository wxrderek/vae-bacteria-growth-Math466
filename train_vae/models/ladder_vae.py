import torch
import torch.nn as nn
import torch.nn.functional as F

class LadderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels*2, 3, padding=1)
        )
        self.skip = nn.Conv1d(in_channels, out_channels*2, 1)

    def forward(self, x):
        return self.conv(x) + self.skip(x)

class LadderVAE(nn.Module):
    def __init__(self, latent_dims=[32, 16, 8], seq_length=300):
        super().__init__()
        self.seq_length = seq_length
        self.encoder = nn.ModuleList([
            LadderBlock(1, 32),
            LadderBlock(64, 16),
            LadderBlock(32, 8)
        ])
        
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(16, 32, 3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose1d(32, 64, 3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose1d(64, 1, 3, padding=1)
            )
        ])
        
        self.fc_mu = nn.ModuleList([
            nn.Linear(64*seq_length, int(latent_dims[0])),
            nn.Linear(32*seq_length//2, int(latent_dims[1])),
            nn.Linear(16*seq_length//4, int(latent_dims[2]))
        ])
        
        self.fc_logvar = nn.ModuleList([
            nn.Linear(64*seq_length, int(latent_dims[0])),
            nn.Linear(32*seq_length//2, int(latent_dims[1])),
            nn.Linear(16*seq_length//4, int(latent_dims[2]))
        ])

        self.fc_decode = nn.ModuleList([
            nn.Linear(latent_dims[2], 16 * (seq_length // 4)),  # for z3
            nn.Linear(latent_dims[1], 32 * (seq_length // 2)),  # for z2
            nn.Linear(latent_dims[0], 64 * seq_length)          # for z1
        ])

    def encode(self, x):
        mus, logvars = [], []
        for i, block in enumerate(self.encoder):
            x = block(x)
            mu = self.fc_mu[i](x.flatten(1))
            logvar = self.fc_logvar[i](x.flatten(1))
            mus.append(mu)
            logvars.append(logvar)
            x = F.max_pool1d(x, 2)
        return mus[::-1], logvars[::-1]  # Reverse for decoder

    def decode(self, zs):
        shapes = [
            (16, self.seq_length // 4),  # for z3
            (32, self.seq_length // 2),  # for z2
            (64, self.seq_length)        # for z1
        ]
        x = None
        for i, (z, fc, block, shape) in enumerate(zip(zs, self.fc_decode, self.decoder, shapes)):
            x_proj = fc(z).view(-1, *shape)
            if x is None:
                x = x_proj
            else:
                if x.shape[-1] != x_proj.shape[-1]:
                    x = F.interpolate(x, size=x_proj.shape[-1])
                x = x + x_proj
            x = block(x)
            if i < len(self.decoder) - 1:
                x = F.interpolate(x, scale_factor=2)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        means, logvars = self.encode(x)
        zs = [self.reparameterize(mu, logvar) for mu, logvar in zip(means, logvars)]
        reconstruction = self.decode(zs)
        return reconstruction, means, logvars, zs