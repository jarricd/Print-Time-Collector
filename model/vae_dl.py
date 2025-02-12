import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_size),
            nn.ReLU(),
        )
        self.dist_mean_layer = nn.Linear(hidden_size, latent_size)
        self.log_var_layer = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        layer_output = self.forward(x)
        distribution_mean = self.dist_mean_layer(layer_output)
        distribution_log_variance = self.log_var_layer(layer_output)
        return distribution_mean, distribution_log_variance


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, latent_space):
        reconstructed_x = self.layers.forward(latent_space)
        return reconstructed_x

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(input_size, hidden_size, latent_size)

    def reparameterize(self, mean, std):
        eps = torch.radn_like(std)
        return mean + std * eps

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)

if __name__ == "__main__":
    hyper_params_dict = {
        "latent_size": 256,
        "hidden_layer_size": 512,
    }
