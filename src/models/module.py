from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embedding import FourierFeatMapping, IdentityMapping, PositionalEncoding
from models.init import *


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


# Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
# special first-layer initialization scheme
NONLINEARITY_AND_INIT = {
    'sine': (Sine(), sine_init, first_layer_sine_init),
    'relu': (nn.ReLU(inplace=True), init_weights_relu, None),
    'lrelu': (nn.LeakyReLU(inplace=True), init_weights_relu, None),
    'sigmoid': (nn.Sigmoid(), init_weights_xavier, None),
    'tanh': (nn.Tanh(), init_weights_xavier, None),
    'selu': (nn.SELU(inplace=True), init_weights_selu, None),
    'softplus': (nn.Softplus(), init_weights_normal, None),
    'elu': (nn.ELU(inplace=True), init_weights_elu, None),
    'identity': (nn.Identity(), None, None)
}

""" MLP Modules
"""


class FCBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_norm: bool = True,
        nonlinearity: str = 'relu'
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.residual = (in_features == out_features)  # when inputs and outputs have the same dimension, build a residual block
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.nonlinearity = NONLINEARITY_AND_INIT[nonlinearity][0]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensors of shape (..., in_features).

        Returns:
            (torch.Tensor): Tensors after linear transformation ($y = xA^T + b$) of shape (..., out_features).
        """
        out = self.linear(x)

        if self.layer_norm:
            out = self.layer_norm(out)

        if self.residual:
            out = self.nonlinearity(out + x)
        else:
            out = self.nonlinearity(out)

        return out


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: int,
        hidden_features: int,
        layer_norm: bool = True,
        skip_connection: bool = True,
        outermost_linear: bool = False,
        nonlinearity: str = 'relu',
        weight_init=None
    ):
        super().__init__()

        nl, nl_weight_init, first_layer_init = NONLINEARITY_AND_INIT[nonlinearity]
        if weight_init is None:
            weight_init = nl_weight_init

        self.skip_connection = skip_connection

        self.layers = nn.ModuleList()
        self.layers.append(FCBlock(in_features, hidden_features, layer_norm, nonlinearity=nonlinearity))
        for _ in range(hidden_layers):
            self.layers.append(FCBlock(in_features + hidden_features if skip_connection else hidden_features, hidden_features, layer_norm, nonlinearity=nonlinearity))
        if outermost_linear:
            self.layers.append(FCBlock(hidden_features, out_features, layer_norm=False, nonlinearity='identity'))
        else:
            self.layers.append(FCBlock(hidden_features, out_features, layer_norm=False, nonlinearity=nonlinearity))

        self.layers.apply(weight_init)
        if first_layer_init is not None:  # apply special initialization to first layer, if applicable
            self.layers[0].apply(first_layer_init)

    def forward(self, x):
        out = self.layers[0](x)
        for i in range(1, len(self.layers) - 1):
            if self.skip_connection:
                out = self.layers[i](torch.cat((out, x), dim=-1))
            else:
                out = self.layers[i](out)
        out = self.layers[-1](out)

        return out


class StrandEncoder(MLP):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: int,
        hidden_features: int,
        layer_norm: bool = True,
        skip_connection: bool = True,
        nonlinearity: str = 'relu',
        weight_init=None
    ):
        super().__init__(in_features, hidden_features, hidden_layers - 1, hidden_features, layer_norm, skip_connection, outermost_linear=False, nonlinearity=nonlinearity, weight_init=weight_init)
        self.mu = nn.Linear(hidden_features, out_features)
        self.log_sigma = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        out = super().forward(x)
        return self.mu(out), self.log_sigma(out)


""" ModSIREN Modules
"""


class SynthesisMLP(MLP):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: int,
        hidden_features: int,
        layer_norm: bool = True,
        skip_connection: bool = True,
        outermost_linear: bool = False,
        nonlinearity: str = 'relu',
        weight_init=None
    ):
        super().__init__(in_features, out_features, hidden_layers, hidden_features, layer_norm, skip_connection, outermost_linear, nonlinearity, weight_init)

    def forward(self, x, gating_layers=None):
        out = self.layers[0](x)
        for i in range(1, len(self.layers) - 1):
            if self.skip_connection:
                out = self.layers[i](torch.cat((out, x), dim=-1))
            else:
                out = self.layers[i](out)
            if gating_layers:
                out = gating_layers[i - 1] * out
        out = self.layers[-1](out)

        return out


class ModulatorMLP(MLP):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: int,
        hidden_features: int,
        layer_norm: bool = True,
        skip_connection: bool = True,
        outermost_linear: bool = False,
        nonlinearity: str = 'relu',
        weight_init=None
    ):
        super().__init__(in_features, out_features, hidden_layers, hidden_features, layer_norm, skip_connection, outermost_linear, nonlinearity, weight_init)

    def forward(self, x):
        out = self.layers[0](x)
        gating_layers = []
        for i in range(1, len(self.layers) - 1):
            if self.skip_connection:
                out = self.layers[i](torch.cat((out, x), dim=-1))
            else:
                out = self.layers[i](out)
            gating_layers.append(out)

        return gating_layers


class ModSIREN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: int,
        hidden_features: int,
        latent_dim: int,
        concat: bool = True,
        synthesis_layer_norm: bool = True,
        modulator_layer_norm: bool = True,
        synthesis_nonlinearity: Optional[str] = 'relu',
        modulator_nonlinearity: Optional[str] = 'relu',
        pos_embed: str = 'identity',
        num_freqs: int = 7,
        freq_scale: float = 1.
    ):
        super().__init__()

        if pos_embed == 'ffm':
            self.pos_embed = FourierFeatMapping(in_features, num_freqs, freq_scale)
        elif pos_embed == 'pe':
            self.pos_embed = PositionalEncoding(in_features, num_freqs)
        else:
            self.pos_embed = IdentityMapping(in_features)
        in_features = self.pos_embed.out_dim

        self.concat = concat

        if modulator_nonlinearity:
            self.synthesis = SynthesisMLP(in_features, out_features, hidden_layers, hidden_features, synthesis_layer_norm,
                                          skip_connection=True, outermost_linear=True, nonlinearity=synthesis_nonlinearity)
            if concat:
                self.modulator = ModulatorMLP(latent_dim + in_features, out_features, hidden_layers, hidden_features, modulator_layer_norm,
                                              skip_connection=True, outermost_linear=True, nonlinearity=modulator_nonlinearity)
            else:
                self.modulator = ModulatorMLP(latent_dim, out_features, hidden_layers, hidden_features, modulator_layer_norm,
                                              skip_connection=True, outermost_linear=True, nonlinearity=modulator_nonlinearity)
        else:
            self.synthesis = SynthesisMLP(latent_dim + in_features, out_features, hidden_layers, hidden_features, synthesis_layer_norm,
                                          skip_connection=True, outermost_linear=True, nonlinearity=synthesis_nonlinearity)
            self.modulator = None

    def forward(self, coords, embedding=None):
        coords = self.pos_embed(coords)

        if embedding is not None:
            if embedding.ndim != coords.ndim:
                embedding = embedding.unsqueeze(1).expand(-1, coords.shape[1], -1)

            gating_layers = None
            if self.modulator:
                if self.concat:
                    gating_layers = self.modulator(torch.cat([embedding, coords], dim=-1))
                else:
                    gating_layers = self.modulator(embedding)
                out = self.synthesis(coords, gating_layers)
            else:
                out = self.synthesis(torch.cat([embedding, coords], dim=-1), gating_layers=None)
        else:
            out = self.synthesis(coords)

        return out


""" CNN Modules
"""


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        batch_norm: bool,
        nonlinearity: str
    ):
        super().__init__()

        self.nonlinearity = NONLINEARITY_AND_INIT[nonlinearity][0]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            self.nonlinearity
        )

    def forward(self, x):
        return self.conv(x)


class ConvTranspose2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        batch_norm: bool,
        nonlinearity: str
    ):
        super().__init__()

        self.nonlinearity = NONLINEARITY_AND_INIT[nonlinearity][0]
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=1 if stride == 2 else 0),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            self.nonlinearity
        )

    def forward(self, x):
        return self.conv(x)


class TextureEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        batch_norm: bool = False,
        nonlinearity: str = 'relu'
    ):
        super().__init__()

        self.nonlinearity = NONLINEARITY_AND_INIT[nonlinearity][0]
        # self.channels_list = [2048, 2048, 512, 2048, 2048, 512, 2048, 2048, 512, 1024]
        self.channels_list = [1024, 1024, 512, 1024, 1024, 512, 1024, 1024, 512, 1024]  # NOTE: reduce hidden dimensions to match parameter numbers reported in GroomGen
        self.downsample_layers = [1, 4, 7]
        self.skip_connection = [2, 5, 8]

        self.conv_layers = nn.ModuleList()
        num_layers = len(self.channels_list)
        for i in range(num_layers):
            if i == 0:
                in_channels = in_features
                out_channels = self.channels_list[i]
            else:
                in_channels = self.channels_list[i - 1]
                out_channels = self.channels_list[i]
            if i in self.downsample_layers:
                kernel_size = 3
                stride = 2
                padding = 1
            elif i == 9:
                kernel_size = 4
                stride = 1
                padding = 0
            else:
                kernel_size = 1
                stride = 1
                padding = 0
            self.conv_layers.append(Conv2dBlock(in_channels, out_channels, kernel_size, stride, padding, batch_norm=batch_norm, nonlinearity=nonlinearity))
        self.conv_layers.append(nn.Conv2d(self.channels_list[-1], out_features, 1, 1, 0))
        self.shortcut = nn.Conv2d(in_features, 512, 1, 1, 0)

    def forward(self, x):
        out = x
        skip = self.shortcut(F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False))
        for i, layer in enumerate(self.conv_layers):
            out = layer(out)
            if i in self.skip_connection:
                out = out + skip
                skip = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)
        return out


class TextureDecoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        batch_norm: bool = False,
        nonlinearity: str = 'relu'
    ):
        super().__init__()

        self.nonlinearity = NONLINEARITY_AND_INIT[nonlinearity][0]
        # self.channels_list = [1024, 512, 2048, 2048, 512, 2048, 2048, 512, 2048, 2048]
        self.channels_list = [1024, 512, 1024, 1024, 512, 1024, 1024, 512, 1024, 1024]  # NOTE: reduce hidden dimensions to match parameter numbers reported in GroomGen
        self.upsample_layers = [3, 6, 9]
        self.skip_connection = [1, 4, 7]

        self.conv_layers = nn.ModuleList()
        num_layers = len(self.channels_list)
        for i in range(num_layers):
            if i == 0:
                in_channels = in_features
                out_channels = self.channels_list[i]
            else:
                in_channels = self.channels_list[i - 1]
                out_channels = self.channels_list[i]
            if i in self.upsample_layers:
                kernel_size = 3
                stride = 2
                padding = 1
            elif i == 1:
                kernel_size = 4
                stride = 1
                padding = 0
            else:
                kernel_size = 1
                stride = 1
                padding = 0
            self.conv_layers.append(ConvTranspose2dBlock(in_channels, out_channels, kernel_size, stride, padding, batch_norm=batch_norm, nonlinearity=nonlinearity))
        self.conv_layers.append(nn.Conv2d(self.channels_list[-1], out_features, 1, 1, 0))

    def forward(self, x):
        out = x
        skip = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        for i, layer in enumerate(self.conv_layers):
            out = layer(out)
            if i in self.skip_connection:
                out = out + skip
                skip = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        return out


class Generator(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_layers: int,
        hidden_features: int,
        batch_norm: bool = False,
        nonlinearity: str = 'relu'
    ):
        super().__init__()

        self.nonlinearity = NONLINEARITY_AND_INIT[nonlinearity][0]
        self.skip_connection = [2, 5, 8, 10]

        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_channels = in_features
                out_channels = hidden_features
            else:
                in_channels = hidden_features
                out_channels = hidden_features
            if i in [1, 4, 7, 10]:
                kernel_size = 13
            else:
                kernel_size = 1
            padding = (kernel_size - 1) // 2
            self.conv_layers.append(Conv2dBlock(in_channels, out_channels, kernel_size, 1, padding, batch_norm=batch_norm, nonlinearity=nonlinearity))
        self.conv_layers.append(nn.Conv2d(hidden_features, out_features, 1, 1, 0))

    def forward(self, x):
        # print(f'x: {x.shape}')
        out = x
        skip = None
        for i, layer in enumerate(self.conv_layers):
            out = layer(out)
            # print(f'out: {out.shape}')
            if i in self.skip_connection:
                if skip is not None:
                    out = out + skip
                skip = out
                # print(f'skip: {skip.shape}')
        return out


class Discriminator(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_layers: int,
        hidden_features: int,
        batch_norm: bool = False,
        nonlinearity: str = 'relu'
    ):
        super().__init__()

        self.nonlinearity = NONLINEARITY_AND_INIT[nonlinearity][0]
        self.skip_connection = [2, 5, 8, 10]

        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_channels = in_features
                out_channels = hidden_features
            else:
                in_channels = hidden_features
                out_channels = hidden_features
            if i in [1, 4, 7, 10]:
                kernel_size = 13
            else:
                kernel_size = 1
            padding = (kernel_size - 1) // 2
            self.conv_layers.append(Conv2dBlock(in_channels, out_channels, kernel_size, 1, padding, batch_norm=batch_norm, nonlinearity=nonlinearity))
        self.conv_layers.append(nn.Conv2d(hidden_features, out_features, 1, 1, 0))

    def forward(self, x):
        out = x
        skip = None
        for i, layer in enumerate(self.conv_layers):
            out = layer(out)
            if i in self.skip_connection:
                if skip is not None:
                    out = out + skip
                skip = out
        return out
