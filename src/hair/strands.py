from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from hair.rotational_repr import cartesian_to_rotational_repr, dot, forward_kinematics, integrate_strand_position, rotation_between_vectors
from utils.misc import EPSILON
from utils.rotation import rotation_6d_to_matrix


@dataclass
class Strands:
    position: Optional[torch.Tensor] = None
    """ 3D position on each point of the strand """

    rotation: Optional[torch.Tensor] = None
    """ Global rotation on each edge of the strand (either 6D rotation or rotation matrix) """

    length: Optional[torch.Tensor] = None
    """ Length on each edge of the strand """

    def __len__(self) -> int:
        """
        Returns:
            (int): Number of strands in the pack.
        """
        for f in fields(self):
            attr = getattr(self, f.name)
            if attr is not None:
                return attr.shape[0]

        raise AttributeError('Empty Strands class does not have attribute `__len__`')

    @property
    def shape(self) -> Tuple[int]:
        """
        Returns:
            (int): Shape of the strands pack.
        """
        if self.position is not None:
            return self.position.shape[:-2]
        if self.rotation is not None:
            return self.rotation.shape[:-2] if self.rotation.shape[-1] == 6 else self.rotation.shape[:-3]
        if self.length is not None:
            return self.length.shape[:-1]
        else:
            raise AttributeError('Empty Strands class does not have attribute `shape`')

    @property
    def ndim(self) -> int:
        """
        Returns:
            (int): Number of spatial dimensions for the strands pack.
        """
        if self.position is not None:
            return self.position.ndim - 2
        if self.rotation is not None:
            return self.rotation.ndim - 2 if self.rotation.shape[-1] == 6 else self.rotation.ndim - 3
        if self.length is not None:
            return self.length.ndim - 1
        else:
            raise AttributeError('Empty Strands class does not have attribute `ndim`')

    def _apply(self, fn) -> Strands:
        """ Apply the function `fn` on each of the channels, if not None.
            Returns a new instance with the processed channels.
        """
        data = {}
        for f in fields(self):
            attr = getattr(self, f.name)
            data[f.name] = None if attr is None else fn(attr)
        return Strands(**data)

    @staticmethod
    def _apply_on_list(lst, fn) -> Strands:
        """ Applies the function `fn` on each entry in the list `lst`.
            Returns a new instance with the processed channels.
        """
        data = {}
        for l in lst:  # gather the property of all entries into a dict
            for f in fields(l):
                attr = getattr(l, f.name)
                if f.name not in data:
                    data[f.name] = []
                data[f.name].append(attr)
        for k, v in data.items():
            data[k] = None if None in v else fn(v)
        return Strands(**data)

    @classmethod
    def cat(cls, lst: List[Strands], dim: int = 0) -> Strands:
        """ Concatenate multiple strands into a single Strands object.

        Args:
            lst (List[Strands]): List of strands to concatenate, expected to have the same spatial dimensions, except of dimension dim.
            dim (int): Spatial dimension along which the concatenation should take place.

        Returns:
            (Strands): A single Strands object with the concatenation of given strands packs.
        """
        if dim < 0:
            dim -= 1
        if dim > lst[0].ndim - 1 or dim < -lst[0].ndim:
            raise IndexError(f"Dimension out of range (expected to be in range of [{-lst[0].ndim}, {lst[0].ndim-1}], but got {dim})")

        return Strands._apply_on_list(lst, lambda x: torch.cat(x, dim=dim))

    @classmethod
    def stack(cls, lst: List[Strands], dim: int = 0) -> Strands:
        """ Stack multiple strands into a single Strands object.

        Args:
            lst (List[Strands]): List of strands to stack, expected to have the same spatial dimensions.
            dim (int): Spatial dimension along which the stack should take place.

        Returns:
            (Strands): A single Strands object with the stacked strands packs.
        """
        return Strands._apply_on_list(lst, lambda x: torch.stack(x, dim=dim))

    def __getitem__(self, idx) -> Strands:
        """ Get strand on the index `idx`. """
        return self._apply(lambda x: x[idx])

    def reshape(self, *dims: Tuple[int]) -> Strands:
        """ Reshape strands to the given `dims`. """
        def _reshape(x):
            extra_dims = self.ndim - x.ndim
            return x.reshape(*dims + x.shape[extra_dims:])
        return self._apply(_reshape)

    def index_select(self, dim: int, index: torch.Tensor) -> Strands:
        """ Index strands along dimension `dim` using the entries in `index`. """
        return self._apply(lambda x: x.index_select(dim, index))

    def squeeze(self, dim: int) -> Strands:
        """ Squeeze strands on dimension `dim`. """
        return self._apply(lambda x: x.squeeze(dim))

    def unsqueeze(self, dim: int) -> Strands:
        """ Unsqueeze strands on dimension `dim`. """
        return self._apply(lambda x: x.unsqueeze(dim))

    def contiguous(self) -> Strands:
        """ Force strands to reside on a contiguous memory. """
        return self._apply(lambda x: x.contiguous())

    def to(self, *args, **kwargs) -> Strands:
        """ Shift strands to a different device / dtype. """
        return self._apply(lambda x: x.to(*args, **kwargs))

    @staticmethod
    def _low_pass_filter(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """ Apply low-pass filtering on the input (implemented as 1D convolution with moving average kernels). """
        half = int(kernel_size // 2)
        x_conv = x.clone()

        start_idx = 1
        for i in range(start_idx, x_conv.shape[-2] - 1):
            i0, i1 = i - half, i + half
            window = torch.zeros_like(x[..., i, :])
            for j in range(i0, i1 + 1):
                if j < 0:
                    p = 2.0 * x[..., 0, :] - x[..., -j, :]
                elif j >= x.shape[-2]:
                    p = 2.0 * x[..., -1, :] - x[..., x.shape[-2] - j - 2, :]
                else:
                    p = x[..., j, :]
                window += p
            x_conv[..., i, :] = window / kernel_size

        return x_conv

    def smooth(self, kernel_size: int) -> Strands:
        """ Smooth strands with a low-pass filter. """
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        position = Strands._low_pass_filter(self.position, kernel_size)

        return Strands(position=position)

    def filter(self, index: torch.Tensor) -> Strands:
        """ Filter strands according to the given index. """
        position = self.position.clone()
        position[index] = 0

        return Strands(position=position)

    def to_canonical(self, guide_strands: Strands) -> Strands:
        """ Transform strands into canonical space (transformation $\phi$ in the paper). """
        if guide_strands.rotation is None or guide_strands.length is None:
            rotation, length = cartesian_to_rotational_repr(guide_strands.position, global_rot=True)
            guide_strands.rotation = rotation
            guide_strands.length = length

        if guide_strands.rotation.shape[-1] == 6:
            guide_strands.rotation = rotation_6d_to_matrix(guide_strands.rotation)

        segment = self.position[..., 1:, :] - self.position[..., :-1, :]
        segment = torch.matmul(torch.linalg.inv(guide_strands.rotation), segment[..., None]).squeeze(-1)  # eliminate rotation from guide strands
        forward = torch.zeros_like(segment)
        forward[..., 1] = -1

        position = integrate_strand_position(segment)
        if self.rotation is None:
            rotation = rotation_between_vectors(forward, F.normalize(segment, dim=-1))
        else:
            if self.rotation.shape[-1] == 6:
                self.rotation = rotation_6d_to_matrix(self.rotation)
            rotation = torch.matmul(torch.linalg.inv(guide_strands.rotation), self.rotation)
        if self.length is None:
            length = torch.norm(segment, dim=-1)
        else:
            length = self.length

        return Strands(position=position,
                       rotation=rotation,
                       length=length
                       )

    def to_world(self, guide_strands: Strands) -> Strands:
        """ Transform strands into world space (transformation $\phi^{-1}$ in the paper). """
        if guide_strands.rotation is None or guide_strands.length is None:
            rotation, length = cartesian_to_rotational_repr(guide_strands.position, global_rot=True)
            guide_strands.rotation = rotation
            guide_strands.length = length

        if guide_strands.rotation.shape[-1] == 6:
            guide_strands.rotation = rotation_6d_to_matrix(guide_strands.rotation)

        segment = self.position[..., 1:, :] - self.position[..., :-1, :]
        segment = torch.matmul(guide_strands.rotation, segment[..., None]).squeeze(-1)
        forward = torch.zeros_like(segment)
        forward[..., 1] = -1

        position = integrate_strand_position(segment)
        if self.rotation is None:
            rotation = rotation_between_vectors(forward, F.normalize(segment, dim=-1))
        else:
            if self.rotation.shape[-1] == 6:
                self.rotation = rotation_6d_to_matrix(self.rotation)
            rotation = torch.matmul(guide_strands.rotation, self.rotation)
        if self.length is None:
            length = torch.norm(segment, dim=-1)
        else:
            length = self.length

        return Strands(position=position,
                       rotation=rotation,
                       length=length
                       )

    # def to_canonical(self, guide_strands: Strands, nonlinear: bool = True) -> Strands:
    #     """ Transform strands into canonical space (transformation $\phi$ in the paper). """
    #     if guide_strands.rotation is None or guide_strands.length is None:
    #         rotation, length = cartesian_to_rotational_repr(guide_strands.position, global_rot=True)
    #         guide_strands.rotation = rotation
    #         guide_strands.length = length

    #     if guide_strands.rotation.shape[-1] == 6:
    #         guide_strands.rotation = rotation_6d_to_matrix(guide_strands.rotation)

    #     segment = self.position[..., 1:, :] - self.position[..., :-1, :]
    #     segment = torch.matmul(torch.linalg.inv(guide_strands.rotation), segment[..., None]).squeeze(-1)  # eliminate rotation from guide strands
    #     forward = torch.zeros_like(segment)
    #     forward[..., 1] = -1
    #     cosine = dot(forward, F.normalize(segment, dim=-1)).abs()

    #     S = (torch.norm(segment, dim=-1) * cosine).sum(dim=-1, keepdim=True)
    #     G = guide_strands.length.sum(dim=-1, keepdim=True)
    #     if nonlinear:
    #         a = (S.pow(2) - G.pow(2)) / (2 * G + EPSILON)
    #         S_c = torch.where(S > G + EPSILON, torch.sqrt(2 * a + 1), S / (G + EPSILON))
    #     else:
    #         S_c = S / (G + EPSILON)

    #     scaling = torch.zeros_like(guide_strands.rotation)
    #     scaling[..., 0, 0] = 1
    #     scaling[..., 1, 1] = S_c / (S + EPSILON)
    #     scaling[..., 2, 2] = 1
    #     segment = torch.matmul(scaling, segment[..., None]).squeeze(-1)

    #     position = integrate_strand_position(segment)
    #     rotation = rotation_between_vectors(forward, F.normalize(segment, dim=-1))
    #     length = torch.norm(segment, dim=-1)

    #     return Strands(position=position,
    #                    rotation=rotation,
    #                    length=length
    #                    )

    # def to_world(self, guide_strands: Strands, nonlinear: bool = True) -> Strands:
    #     """ Transform strands into world space (transformation $\phi^{-1}$ in the paper). """
    #     segment = self.position[..., 1:, :] - self.position[..., :-1, :]
    #     forward = torch.zeros_like(segment)
    #     forward[..., 1] = -1
    #     cosine = dot(forward, F.normalize(segment, dim=-1)).abs()

    #     if self.length is None:
    #         self.length = torch.norm(segment, dim=-1)
    #     if guide_strands.rotation is None or guide_strands.length is None:
    #         guide_segment = guide_strands.position[..., 1:, :] - guide_strands.position[..., :-1, :]
    #         guide_strands.rotation = rotation_between_vectors(forward, F.normalize(guide_segment, dim=-1))
    #         guide_strands.length = torch.norm(guide_segment, dim=-1)

    #     if guide_strands.rotation.shape[-1] == 6:
    #         guide_strands.rotation = rotation_6d_to_matrix(guide_strands.rotation)

    #     S_c = (self.length * cosine).sum(dim=-1, keepdim=True)
    #     G = guide_strands.length.sum(dim=-1, keepdim=True)
    #     if nonlinear:
    #         a = (S_c.pow(2) - 1) / 2
    #         S = torch.where(S_c > 1 + EPSILON, torch.sqrt((G + a).pow(2) - a.pow(2) + EPSILON), S_c * G)
    #     else:
    #         S = S_c * G

    #     scaling = torch.zeros_like(guide_strands.rotation)
    #     scaling[..., 0, 0] = 1
    #     scaling[..., 1, 1] = S / (S_c + EPSILON)
    #     scaling[..., 2, 2] = 1
    #     transform = torch.matmul(guide_strands.rotation, scaling)
    #     segment = torch.matmul(transform, segment[..., None]).squeeze(-1)

    #     position = integrate_strand_position(segment)
    #     rotation = rotation_between_vectors(forward, F.normalize(segment, dim=-1))
    #     length = torch.norm(segment, dim=-1)

    #     return Strands(position=position,
    #                    rotation=rotation,
    #                    length=length
    #                    )

    # def to_canonical(self, guide_strands: Strands, smooth: bool = True) -> Strands:
    #     """ Transform strands into canonical space (transformation $\phi$ in the paper). """
    #     if smooth:
    #         smoothed_strands = self.smooth(kernel_size=self.position.shape[-2] // 4)
    #         residual = self.position - smoothed_strands.position
    #     else:
    #         smoothed_strands = Strands(position=self.position)
    #         residual = None

    #     if guide_strands.rotation is None or guide_strands.length is None:
    #         rotation, length = cartesian_to_rotational_repr(guide_strands.position, global_rot=True)
    #         guide_strands.rotation = rotation
    #         guide_strands.length = length

    #     if guide_strands.rotation.shape[-1] == 6:
    #         guide_strands.rotation = rotation_6d_to_matrix(guide_strands.rotation)

    #     segment = smoothed_strands.position[..., 1:, :] - smoothed_strands.position[..., :-1, :]
    #     segment = torch.matmul(torch.linalg.inv(guide_strands.rotation), segment[..., None]).squeeze(-1)  # eliminate rotation from guide strands
    #     forward = torch.zeros_like(segment)
    #     forward[..., 1] = -1
    #     cosine = dot(forward, F.normalize(segment, dim=-1)).abs()

    #     S = (torch.norm(segment, dim=-1) * cosine).sum(dim=-1, keepdim=True)
    #     G = guide_strands.length.sum(dim=-1, keepdim=True)
    #     a = (S.pow(2) - G.pow(2)) / (2 * G + EPSILON)
    #     S_c = torch.where(S > G + EPSILON, torch.sqrt(2 * a + 1), S / (G + EPSILON))

    #     scaling = torch.zeros_like(guide_strands.rotation)
    #     scaling[..., 0, 0] = 1
    #     scaling[..., 1, 1] = S_c / (S + EPSILON)
    #     scaling[..., 2, 2] = 1
    #     segment = torch.matmul(scaling, segment[..., None]).squeeze(-1)

    #     position = integrate_strand_position(segment)
    #     rotation = rotation_between_vectors(forward, F.normalize(segment, dim=-1))
    #     length = torch.norm(segment, dim=-1)

    #     if residual is not None:
    #         transform = torch.matmul(rotation, torch.linalg.inv(smoothed_strands.rotation))
    #         residual = torch.matmul(transform[..., :-1, :, :], residual[..., 1:-1, :, None]).squeeze(-1)

    #     return Strands(position=position,
    #                    rotation=rotation,
    #                    length=length,
    #                    residual=residual
    #                    )

    # def to_world(self, guide_strands: Strands) -> Strands:
    #     """ Transform strands into world space (transformation $\phi^{-1}$ in the paper). """
    #     segment = self.position[..., 1:, :] - self.position[..., :-1, :]
    #     forward = torch.zeros_like(segment)
    #     forward[..., 1] = -1
    #     cosine = dot(forward, F.normalize(segment, dim=-1)).abs()

    #     if self.rotation is None or self.length is None:
    #         self.rotation = rotation_between_vectors(forward, F.normalize(segment, dim=-1))
    #         self.length = torch.norm(segment, dim=-1)
    #     if guide_strands.rotation is None or guide_strands.length is None:
    #         guide_segment = guide_strands.position[..., 1:, :] - guide_strands.position[..., :-1, :]
    #         guide_strands.rotation = rotation_between_vectors(forward, F.normalize(guide_segment, dim=-1))
    #         guide_strands.length = torch.norm(guide_segment, dim=-1)

    #     if self.rotation.shape[-1] == 6:
    #         self.rotation = rotation_6d_to_matrix(self.rotation)
    #     if guide_strands.rotation.shape[-1] == 6:
    #         guide_strands.rotation = rotation_6d_to_matrix(guide_strands.rotation)

    #     S_c = (self.length * cosine).sum(dim=-1, keepdim=True)
    #     G = guide_strands.length.sum(dim=-1, keepdim=True)
    #     a = (S_c.pow(2) - 1) / 2
    #     S = torch.where(S_c > 1 + EPSILON, torch.sqrt((G + a).pow(2) - a.pow(2) + EPSILON), S_c * G)

    #     scaling = torch.zeros_like(guide_strands.rotation)
    #     scaling[..., 0, 0] = 1
    #     scaling[..., 1, 1] = S / (S_c + EPSILON)
    #     scaling[..., 2, 2] = 1
    #     transform = torch.matmul(guide_strands.rotation, scaling)
    #     segment = torch.matmul(transform, segment[..., None]).squeeze(-1)

    #     position = integrate_strand_position(segment)
    #     rotation = rotation_between_vectors(forward, F.normalize(segment, dim=-1))
    #     length = torch.norm(segment, dim=-1)

    #     if self.residual is not None:
    #         transform = torch.matmul(rotation, torch.linalg.inv(self.rotation))
    #         residual = torch.matmul(transform[..., :-1, :, :], self.residual[..., None]).squeeze(-1)
    #         residual = F.pad(residual, (0, 0, 1, 1), mode='constant', value=0)
    #         position = position + residual

    #     return Strands(position=position,
    #                    rotation=rotation,
    #                    length=length
    #                    )

    # def canonicalize(self, guide_strands: Strands) -> Strands:
    #     """ Transform strands into canonical space (transformation $\phi$ in the paper). """
    #     if self.position.shape[-2] == self.length.shape[-1]:
    #         self.position = F.pad(self.position, (0, 0, 1, 0), mode='constant', value=0)
    #     strands_smoothed = self.smoothing(kernel_size=self.position.shape[-2] // 4)

    #     if guide_strands.rotation.shape[-1] == 6:
    #         guide_strands.rotation = rotation_6d_to_matrix(guide_strands.rotation)
    #     rotation = torch.matmul(torch.linalg.inv(guide_strands.rotation), strands_smoothed.rotation)
    #     S = strands_smoothed.length.sum(dim=-1, keepdim=True)
    #     G = guide_strands.length.sum(dim=-1, keepdim=True)
    #     # k1 = (S.pow(2) - G.pow(2)) / (2 * G)  # S > G
    #     # k2 = (G.pow(2) - S.pow(2)) / (2 * S)  # S <= G
    #     # S_c = torch.where(S > G, torch.sqrt(2 * k1 + 1), torch.sqrt(k2.pow(2) + 1) - k2)
    #     k = (S.pow(2) - G.pow(2)) / (2 * G)  # S > G
    #     S_c = torch.where(S > G, torch.sqrt(2 * k + 1), S / G)
    #     # print(f'S: {S}')
    #     # print(f'G: {G}')
    #     # print(f'S_c: {S_c}')
    #     # if torch.isnan(S_c).any():
    #     #     print('asd')
    #     #     exit(-1)
    #     length = S_c / S * strands_smoothed.length
    #     position = forward_kinematics(rotation, length)

    #     residual = self.position - strands_smoothed.position
    #     residual = torch.matmul(torch.linalg.inv(guide_strands.rotation), residual[..., 1:, :, None]).squeeze(-1)

    #     return Strands(position=position,
    #                    rotation=rotation,
    #                    length=length,
    #                    residual=residual
    #                    )

    # def globalize(self, guide_strands: Strands) -> Strands:
    #     """ Transform strands into global space (transformation $\phi^{-1}$ in the paper). """
    #     if self.rotation.shape[-1] == 6:
    #         self.rotation = rotation_6d_to_matrix(self.rotation)
    #     if guide_strands.rotation.shape[-1] == 6:
    #         guide_strands.rotation = rotation_6d_to_matrix(guide_strands.rotation)

    #     rotation = torch.matmul(guide_strands.rotation, self.rotation)
    #     S_c = self.length.sum(dim=-1, keepdim=True)
    #     G = guide_strands.length.sum(dim=-1, keepdim=True)
    #     # k1 = (S_c.pow(2) - 1) / 2  # S_c > 1
    #     # k2 = (1 - S_c.pow(2)) / (2 * S_c)  # S_c <= 1
    #     # S = torch.where(S_c > 1, torch.sqrt((G + k1).pow(2) - k1.pow(2)), torch.sqrt(G.pow(2) + k2.pow(2)) - k2)
    #     k = (S_c.pow(2) - 1) / 2  # S_c > 1
    #     S = torch.where(S_c > 1, torch.sqrt((G + k).pow(2) - k.pow(2)), S_c * G)
    #     # print(f'S_c: {S_c}')
    #     # print(f'G: {G}')
    #     # print(f'S: {S}')
    #     length = S / S_c * self.length
    #     position = forward_kinematics(rotation, length)

    #     residual = torch.matmul(guide_strands.rotation, self.residual[..., None]).squeeze(-1)
    #     residual = F.pad(residual, (0, 0, 1, 0), mode='constant', value=0)
    #     position = position + residual

    #     return Strands(position=position,
    #                    rotation=rotation,
    #                    length=length
    #                    )

    # def canonicalize(self, guide_strands: Strands) -> Strands:
    #     """ Transform strands into canonical space (transformation $\phi$ in the paper). """
    #     if self.position.shape[-2] == self.length.shape[-1]:
    #         self.position = F.pad(self.position, (0, 0, 1, 0), mode='constant', value=0)
    #     strands_smoothed = self.smoothing(kernel_size=self.position.shape[-2] // 4)

    #     if guide_strands.rotation.shape[-1] == 6:
    #         guide_strands.rotation = rotation_6d_to_matrix(guide_strands.rotation)
    #     rotation = torch.matmul(torch.linalg.inv(guide_strands.rotation), strands_smoothed.rotation)
    #     scale = math.log(2) / torch.log(guide_strands.length.sum(dim=-1, keepdim=True) + 1.0)
    #     length = scale * strands_smoothed.length
    #     position = forward_kinematics(rotation, length)

    #     residual = self.position - strands_smoothed.position
    #     residual = torch.matmul(torch.linalg.inv(guide_strands.rotation), residual[..., 1:, :, None]).squeeze(-1)

    #     return Strands(position=position,
    #                    rotation=rotation,
    #                    length=length,
    #                    residual=residual
    #                    )

    # def globalize(self, guide_strands: Strands) -> Strands:
    #     """ Transform strands into global space (transformation $\phi^{-1}$ in the paper). """
    #     if self.rotation.shape[-1] == 6:
    #         self.rotation = rotation_6d_to_matrix(self.rotation)
    #     if guide_strands.rotation.shape[-1] == 6:
    #         guide_strands.rotation = rotation_6d_to_matrix(guide_strands.rotation)

    #     rotation = torch.matmul(guide_strands.rotation, self.rotation)
    #     scale = torch.log(guide_strands.length.sum(dim=-1, keepdim=True) + 1.0) / math.log(2)
    #     length = scale * self.length
    #     position = forward_kinematics(rotation, length)

    #     residual = torch.matmul(guide_strands.rotation, self.residual[..., None]).squeeze(-1)
    #     residual = F.pad(residual, (0, 0, 1, 0), mode='constant', value=0)
    #     position = position + residual

    #     return Strands(position=position,
    #                    rotation=rotation,
    #                    length=length
    #                    )

    # def canonicalize(self, guide_strands: Strands) -> Strands:
    #     """ Transform strands into canonical space (transformation $\phi$ in the paper). """
    #     if guide_strands.rotation.shape[-1] == 6:
    #         guide_strands.rotation = rotation_6d_to_matrix(guide_strands.rotation)
    #     scaling = torch.zeros_like(guide_strands.rotation)
    #     scaling[..., 0, 0] = 1
    #     scaling[..., 1, 1] = 1 / (guide_strands.length.sum(dim=-1, keepdim=True) + 1e-12)
    #     scaling[..., 2, 2] = 1
    #     transform = torch.matmul(scaling, torch.linalg.inv(guide_strands.rotation))

    #     if self.position.shape[-2] == transform.shape[-3]:
    #         self.position = F.pad(self.position, (0, 0, 1, 0), mode='constant', value=0)
    #     strands_smoothed = self.smoothing(kernel_size=(self.position.shape[-2] - 1) // 2)

    #     segment = strands_smoothed.position[..., 1:, :] - strands_smoothed.position[..., :-1, :]
    #     segment = torch.matmul(transform, segment[..., None]).squeeze(-1)
    #     position = integrate_strand_position(segment)
    #     forward = torch.zeros_like(segment)
    #     forward[..., 1] = -1
    #     rotation = rotation_between_vectors(forward, F.normalize(segment, dim=-1))
    #     length = torch.norm(segment, dim=-1)
    #     residual = self.position - strands_smoothed.position
    #     residual_transform = torch.matmul(rotation, torch.linalg.inv(strands_smoothed.rotation))
    #     residual = torch.matmul(residual_transform, residual[..., 1:, :, None]).squeeze(-1)

    #     return Strands(position=position,
    #                    rotation=rotation,
    #                    length=length,
    #                    residual=residual
    #                    )

    # def globalize(self, guide_strands: Strands) -> Strands:
    #     """ Transform strands into global space (transformation $\phi^{-1}$ in the paper). """
    #     if self.rotation.shape[-1] == 6:
    #         self.rotation = rotation_6d_to_matrix(self.rotation)
    #     if guide_strands.rotation.shape[-1] == 6:
    #         guide_strands.rotation = rotation_6d_to_matrix(guide_strands.rotation)
    #     scaling = torch.zeros_like(guide_strands.rotation)
    #     scaling[..., 0, 0] = 1
    #     scaling[..., 1, 1] = guide_strands.length.sum(dim=-1, keepdim=True) + 1e-12
    #     scaling[..., 2, 2] = 1
    #     transform = torch.matmul(guide_strands.rotation, scaling)

    #     forward = torch.zeros(transform.shape[:-1], device=transform.device)
    #     forward[..., 1] = -1
    #     if self.position is None:
    #         scaling_canonical = torch.zeros_like(self.rotation)
    #         scaling_canonical[..., 0, 0] = self.length
    #         scaling_canonical[..., 1, 1] = self.length
    #         scaling_canonical[..., 2, 2] = self.length
    #         transform_canonical = torch.matmul(scaling_canonical, self.rotation)
    #         segment = torch.matmul(transform_canonical, forward[..., None]).squeeze(-1)
    #     else:
    #         if self.position.shape[-2] == transform.shape[-3]:
    #             self.position = F.pad(self.position, (0, 0, 1, 0), mode='constant', value=0)
    #         segment = self.position[..., 1:, :] - self.position[..., :-1, :]

    #     segment = torch.matmul(transform, segment[..., None]).squeeze(-1)
    #     position = integrate_strand_position(segment)
    #     rotation = rotation_between_vectors(forward, F.normalize(segment, dim=-1))
    #     length = torch.norm(segment, dim=-1)
    #     residual_transform = torch.matmul(rotation, torch.linalg.inv(self.rotation))
    #     residual = torch.matmul(residual_transform, self.residual[..., None]).squeeze(-1)
    #     residual = F.pad(residual, (0, 0, 1, 0), mode='constant', value=0)
    #     position = position + residual

    #     return Strands(position=position,
    #                    rotation=rotation,
    #                    length=length
    #                    )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, strand_repr: str, global_rot: bool = True) -> Strands:
        """ Create a Strands object from torch Tensor with the given representation. """
        if strand_repr == 'rotation':  # 6d rotation + length
            rotation = tensor[..., :6]
            length = torch.abs(tensor[..., 6])
            position = forward_kinematics(rotation, length, global_rot)
            return Strands(position=position, rotation=rotation, length=length)
        elif strand_repr == 'direction':  # direction (finite difference of position)
            position = integrate_strand_position(tensor)
            forward = torch.zeros_like(tensor)
            forward[..., 1] = -1
            rotation = rotation_between_vectors(forward, F.normalize(tensor, dim=-1))
            length = torch.norm(tensor, dim=-1)
            return Strands(position=position, rotation=rotation, length=length)
        elif strand_repr == 'position':  # position
            position = F.pad(tensor, (0, 0, 1, 0), mode='constant', value=0)
            direction = F.normalize(position[..., 1:, :] - position[..., :-1, :], dim=-1)
            forward = torch.zeros_like(direction)
            forward[..., 1] = -1
            rotation = rotation_between_vectors(forward, direction)
            length = torch.norm(position[..., 1:, :] - position[..., :-1, :], dim=-1)
            return Strands(position=position, rotation=rotation, length=length)
        elif strand_repr == 'residual':  # residual
            return Strands(residual=tensor)
        else:
            raise RuntimeError(f'representation {strand_repr} is not supported')

    def to_tensor(self) -> torch.Tensor:
        """ Concatenate strands arrtibutes to a torch Tensor. """
        lst = []
        for f in fields(self):
            attr = getattr(self, f.name)
            if attr is not None:
                lst.append(attr[..., None] if f.name == 'length' else attr)

        return torch.cat(lst, dim=-1)
