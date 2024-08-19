from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from utils.misc import EPSILON


def position_to_frequency(pos: Union[np.ndarray, torch.Tensor], n_bands: int = 33):
    grad = pos[:, 1:] - pos[:, :-1]
    grad = grad.reshape(pos.shape[0], -1, n_bands, 3)
    if isinstance(grad, np.ndarray):
        fourier = np.fft.rfft(grad, n=grad.shape[-2], norm='ortho', axis=-2)
        amp = np.abs(fourier)
    else:
        fourier = torch.fft.rfft(grad, n=grad.shape[-2], dim=-2, norm='ortho')
        amp = torch.abs(fourier)
    cos = fourier.real / (amp + EPSILON)
    sin = fourier.imag / (amp + EPSILON)

    return amp, cos, sin


def euler_integration(grad: torch.Tensor):
    pos = torch.zeros_like(grad)
    pos = F.pad(pos, (0, 0, 1, 0), mode='constant', value=0)
    for i in range(1, pos.shape[-2]):
        pos[..., i, :] = pos[..., i - 1, :] + grad[..., i - 1, :]
    return pos


def frequency_to_position(amp: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, n_bands: int = 33):
    grad = torch.fft.irfft(torch.complex(cos * amp, sin * amp), n=n_bands, norm='ortho', dim=-2)
    # print(f'grad: {grad.shape}')
    grad = grad.reshape(amp.shape[0], -1, 3)
    return euler_integration(grad)
