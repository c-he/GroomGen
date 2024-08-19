import torch
import torch.nn as nn

from models.module import TextureDecoder, TextureEncoder

# ----------------------------------------------------------------------------


class HairstyleVAE(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 latent_dim,
                 variational=True
                 ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.latent_dim = latent_dim
        self.variational = variational

        self.encoder = TextureEncoder(in_features=in_features,
                                      out_features=latent_dim * 2 if variational else latent_dim,
                                      batch_norm=True,
                                      nonlinearity='relu'
                                      )
        self.decoder = TextureDecoder(in_features=latent_dim,
                                      out_features=out_features,
                                      batch_norm=True,
                                      nonlinearity='relu'
                                      )

    def reparameterize(self, mu, log_sigma):
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * log_sigma)

        return mu + eps * std

    def encode(self, batch):
        image = batch['image']
        mask = batch['mask'] * 2 - 1
        x = torch.cat([image, mask], dim=1)
        y = self.encoder(x)
        y = y.flatten(1)
        mu, log_sigma = y[:, :self.latent_dim], y[:, self.latent_dim:]
        if self.variational:
            z = self.reparameterize(mu, log_sigma)
        else:
            z = mu
        return z, mu, log_sigma

    def decode(self, z):
        z = z[..., None, None]
        out = self.decoder(z)
        image = out[:, 1:]
        mask = torch.sigmoid(out[:, :1]) * (1 + 2 * 0.001) - 0.001
        return {'image': image, 'mask': mask}

    def forward(self, x):
        z, _, _ = self.encode(x)
        return self.decode(z)


# ----------------------------------------------------------------------------
