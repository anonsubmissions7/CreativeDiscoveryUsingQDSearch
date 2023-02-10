'''
This File contains the VAE architecture used for majority of the testing.
'''

import torch
import torch.nn.functional as F
# from torchinfo import summary
from torch import nn

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, device):
        super(VariationalEncoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 5, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 5, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 5, stride=2, padding=0),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )

        self.linear1 = nn.Linear(1 * 1 * 1024, latent_dims)
        self.linear2 = nn.Linear(latent_dims, latent_dims)
        self.linear3 = nn.Linear(latent_dims, latent_dims)

        # initialise non model parameters
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0  # stores KL divergence
        self.device = device

        self.mu = None
        self.sigma = None

    def forward(self, x):
        x = x.to(self.device)
        x = self.encoder_conv(x)
        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))

        self.mu = mu
        self.sigma = sigma

        z = mu + sigma * self.N.sample(mu.shape)
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
        # these help make sense of below, although interested in a different method for computing KL divergence
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()

        return z


class VariationalDecoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalDecoder, self).__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, latent_dims),
            nn.LeakyReLU(),
            nn.Linear(latent_dims, 1 * 1* 1024),
            nn.LeakyReLU(),
        )

        # self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1024, 1, 1))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 5, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, 5, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 5, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 5, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, 5, stride=2, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        # x = self.unflatten(x)
        x = torch.reshape(x, (-1, 1024, 1, 1)) # i think 1024 is an error
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)

        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, device):
        super(VariationalAutoencoder, self).__init__()
        self.device = device
        self.encoder = VariationalEncoder(latent_dims, self.device)
        self.decoder = VariationalDecoder(latent_dims)
        self.latent_dims = latent_dims

    def forward(self, x):
        x = x.to(self.device)
        z = self.encoder(x)
        return self.decoder(z)

if __name__ == "__main__":
    model = VariationalAutoencoder(512, device=torch.device("cuda:0"))
    batch_size = 128
    from torchinfo import summary
    summary(model, input_size=(batch_size, 1, 512, 512))
