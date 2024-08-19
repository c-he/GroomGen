import numpy as np
import torch

from utils.fft import frequency_to_position, position_to_frequency

device = torch.device('cuda')
strands = np.random.randn(10, 100, 3)
strands = strands - strands[:, :1].copy()
print(f'strands: {strands.shape}')
amp, cos, sin = position_to_frequency(strands)
print(f'amp: {amp.shape}')
print(f'cos: {cos.shape}')
print(f'sin: {sin.shape}')
amp = torch.tensor(amp, dtype=torch.float32, device=device)
cos = torch.tensor(cos, dtype=torch.float32, device=device)
sin = torch.tensor(sin, dtype=torch.float32, device=device)
pos = frequency_to_position(amp, cos, sin).cpu().numpy()
print(f'pos: {pos.shape}')
print(np.allclose(strands, pos, atol=1e-4))
