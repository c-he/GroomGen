import glob
import math
import os

import numpy as np
from torch.utils.data import Dataset
from utils.misc import load_tensor_dict


class HairStrandsDataset(Dataset):
    def __init__(
        self,
        path,               # Path to pre-computed .npy strand data.
        max_size=None,      # Artificially limit the size of the dataset. None = no limit.
        random_seed=0,      # Random seed to use when applying max_size.
    ):
        self._name = 'hair-strands'  # Name of the dataset.
        self._path = path
        self._all_strands = load_tensor_dict(path)

        # Apply max_size.
        self._raw_idx = np.arange(self._all_strands[self.props[0]].shape[0], dtype=np.int64)
        if (max_size is not None) and self._all_strands[self.props[0]].shape[0] > max_size:
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        return {k: v[raw_idx] for k, v in self._all_strands.items()}

    @property
    def name(self):
        return self._name

    @property
    def props(self):
        return list(self._all_strands.keys())

    @property
    def num_channels(self):
        n_channels = 0
        for key in self._all_strands.keys():
            if key != 'pos':
                n_channels += math.prod(self._all_strands[key].shape[1:])
        return n_channels


class NeuralTextureDataset(Dataset):
    def __init__(
        self,
        path,               # Path to .npz neural textures.
        max_size=None,      # Artificially limit the size of the dataset. None = no limit.
    ):
        self._name = 'hair-neural-textures'  # Name of the dataset.
        self._path = path
        self._img_path = os.path.join(path, 'high-res')
        self._raw_path = os.path.join(path, 'low-res')
        self._all_fnames = [os.path.basename(f) for f in glob.glob(os.path.join(self._img_path, '*.npz'))]
        self._all_fnames = sorted(self._all_fnames)

        self._img_shape = [len(self._all_fnames)] + list(self._load_texture(os.path.join(self._img_path, self._all_fnames[0]))['texture'].shape)
        self._raw_shape = [len(self._all_fnames)] + list(self._load_texture(os.path.join(self._raw_path, self._all_fnames[0]))['texture'].shape)

        # Apply max_size.
        self._raw_idx = np.arange(len(self._all_fnames), dtype=np.int64)
        if (max_size is not None) and len(self._all_fnames) > max_size:
            self._raw_idx = self._raw_idx[:max_size]

    def _load_texture(self, fname):
        return load_tensor_dict(fname)

    def get_raw_texture(self, idx):
        fname = self._all_fnames[self._raw_idx[idx]]
        return self._load_texture(os.path.join(self._raw_path, fname))['texture']

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        fname = self._all_fnames[self._raw_idx[idx]]
        img = self._load_texture(os.path.join(self._img_path, fname))
        raw = self._load_texture(os.path.join(self._raw_path, fname))

        return img['texture'], raw['texture'], img['mask'], raw['mask']

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._name

    @property
    def img_shape(self):
        return list(self._img_shape[1:])

    @property
    def raw_shape(self):
        return list(self._raw_shape[1:])

    @property
    def img_channels(self):
        assert len(self.img_shape) == 3  # CHW
        return self.img_shape[0]

    @property
    def raw_channels(self):
        assert len(self.raw_shape) == 3  # CHW
        return self.raw_shape[0]

    @property
    def img_resolution(self):
        assert len(self.img_shape) == 3  # CHW
        assert self.img_shape[1] == self.img_shape[2]
        return self.img_shape[1]

    @property
    def raw_resolution(self):
        assert len(self.raw_shape) == 3  # CHW
        assert self.raw_shape[1] == self.raw_shape[2]
        return self.raw_shape[1]
