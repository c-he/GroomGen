import numpy as np
import torch
import torch.nn as nn


class IdentityMapping(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    @property
    def out_dim(self):
        return self.in_features

    def forward(self, x):
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, in_features, num_frequencies=10):
        super().__init__()

        self.num_frequencies = num_frequencies
        self.in_features = in_features
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = torch.cat([2**torch.linspace(0, num_frequencies - 1, num_frequencies) * np.pi])

    @property
    def out_dim(self):
        return len(self.funcs) * self.in_features * self.num_frequencies

    def forward(self, x):
        out = []

        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, -1)


class FourierFeatMapping(nn.Module):
    def __init__(self, in_features, num_frequencies=256, scale=10.):
        super().__init__()

        B = torch.normal(0., scale, size=(num_frequencies, in_features))
        self.register_buffer('B', B)

    @property
    def out_dim(self):
        return 2 * self.B.shape[0]

    def forward(self, x):
        x_proj = 2 * np.pi * torch.matmul(x, self.B.T)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
