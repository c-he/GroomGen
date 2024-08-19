import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.fft import euler_integration, frequency_to_position

from models.module import ModSIREN, StrandEncoder

# ----------------------------------------------------------------------------


class StrandVAE(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        latent_dim,
        encoder_layers,
        decoder_layers,
        num_samples,
        variational=True,
        strand_repr="frequency",
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.strand_repr = strand_repr
        self.variational = variational

        self.encoder = StrandEncoder(
            in_features=in_features,
            out_features=latent_dim,
            hidden_layers=encoder_layers - 2,
            hidden_features=hidden_features,
            layer_norm=True,
            skip_connection=True,
            nonlinearity="relu",
        )
        self.decoder = ModSIREN(
            in_features=1,
            out_features=out_features,
            hidden_layers=decoder_layers - 2,
            hidden_features=hidden_features,
            latent_dim=latent_dim,
            concat=False,
            synthesis_layer_norm=True,
            synthesis_nonlinearity="sine",
            modulator_layer_norm=True,
            modulator_nonlinearity="relu",
            pos_embed="identity",
        )

    def reparameterize(self, mu, log_sigma):
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * log_sigma)

        return mu + eps * std

    def encode(self, batch):
        if self.strand_repr == "position":
            x = batch["dir"]
        else:
            x = torch.cat([batch["amp"], batch["cos"], batch["sin"]], dim=-1)
        x = x.flatten(1)
        mu, log_sigma = self.encoder(x)
        if self.variational:
            z = self.reparameterize(mu, log_sigma)
        else:
            z = mu
        return z, mu, log_sigma

    def decode(self, z):
        t = torch.linspace(-1, 1, steps=self.num_samples, device=z.device)[
            None, :, None
        ]
        t = t.expand(z.shape[0], -1, -1)
        y = self.decoder(t, embedding=z)  # (batch_size, num_samples, out_features)
        if self.strand_repr == "position":
            pos = euler_integration(y)
            return {"dir": y, "pos": pos}
        else:
            amp = F.relu(y[..., :3]).reshape(z.shape[0], 3, -1, 3)
            cos = y[..., 3:6].reshape(z.shape[0], 3, -1, 3)
            sin = y[..., 6:].reshape(z.shape[0], 3, -1, 3)
            pos = frequency_to_position(amp, cos, sin, n_bands=33)
            return {"amp": amp, "cos": cos, "sin": sin, "pos": pos}

    def forward(self, x):
        z, _, _ = self.encode(x)
        return self.decode(z)


# ----------------------------------------------------------------------------
