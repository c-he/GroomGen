# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import training_stats
from utils.misc import EPSILON

# ----------------------------------------------------------------------------


class Loss:
    def accumulate_gradients(self, **kwargs):  # to be overridden by subclass
        raise NotImplementedError()

# ----------------------------------------------------------------------------


class StrandVAELoss(Loss):
    def __init__(self, device, model, strand_repr, lambda_kl=0.0):
        super().__init__()
        self.device = device
        self.model = model

        self.strand_repr = strand_repr
        self.lambda_kl = lambda_kl

    def run_model(self, batch):
        z, mu, log_sigma = self.model.module.encode(batch)
        recon = self.model.module.decode(z)
        recon.update({'mu': mu, 'log_sigma': log_sigma})
        return recon

    def kl_loss(self, mu, log_sigma):
        loss = 0.5 * torch.sum(torch.exp(log_sigma) + mu ** 2 - 1 - log_sigma, dim=-1)
        return loss.mean()

    def accumulate_gradients(self, batch):
        with torch.autograd.profiler.record_function('main_forward'):
            gen_output = self.run_model(batch)
            if self.strand_repr == 'direction':
                loss_pos = F.mse_loss(gen_output['pos'], batch['pos'])
                training_stats.report('Loss/position', loss_pos)
                d1 = F.normalize(gen_output['dir'], dim=-1)
                d2 = F.normalize(batch['dir'], dim=-1)
                loss_dir = (1.0 - F.cosine_similarity(d1, d2, dim=-1)).mean()
                training_stats.report('Loss/direction', loss_dir)
                loss_recon = loss_pos + 1e-3 * loss_dir
            else:
                loss_amp = F.l1_loss(gen_output['amp'], batch['amp'])
                training_stats.report('Loss/amplitude', loss_amp)
                amp_weight = batch['amp'] / (batch['amp'].sum(dim=-2, keepdim=True) + EPSILON)
                loss_phase = F.l1_loss(gen_output['cos'], batch['cos'], reduction='none') + F.l1_loss(gen_output['sin'], batch['sin'], reduction='none')
                loss_phase = (loss_phase * amp_weight).mean()
                training_stats.report('Loss/phase', loss_phase)
                loss_recon = loss_amp + loss_phase

            loss_kl = 0.0
            if self.lambda_kl > 0 and self.model.module.variational:
                loss_kl = self.kl_loss(gen_output['mu'], gen_output['log_sigma'])
                training_stats.report('Loss/KL', loss_kl)

            loss = loss_recon + loss_kl * self.lambda_kl
            training_stats.report('Loss/loss', loss)

        with torch.autograd.profiler.record_function('main_backward'):
            loss.backward()

        return loss

# ----------------------------------------------------------------------------


class HairstyleVAELoss(Loss):
    def __init__(self, device, model, lambda_kl=0.0):
        super().__init__()
        self.device = device
        self.model = model

        self.lambda_kl = lambda_kl

    def run_model(self, batch):
        z, mu, log_sigma = self.model.module.encode(batch)
        recon = self.model.module.decode(z)
        recon.update({'mu': mu, 'log_sigma': log_sigma})
        return recon

    def kl_loss(self, mu, log_sigma):
        loss = 0.5 * torch.sum(torch.exp(log_sigma) + mu ** 2 - 1 - log_sigma, dim=-1)
        return loss.mean()

    def accumulate_gradients(self, batch):
        with torch.autograd.profiler.record_function('main_forward'):
            gen_output = self.run_model(batch)
            loss_imag = F.l1_loss(gen_output['image'], batch['image'])
            training_stats.report('Loss/image', loss_imag)
            loss_mask = F.l1_loss(gen_output['mask'], batch['mask'])
            training_stats.report('Loss/mask', loss_mask)
            loss_recon = loss_imag + loss_mask

            loss_kl = 0.0
            if self.lambda_kl > 0 and self.model.module.variational:
                loss_kl = self.kl_loss(gen_output['mu'], gen_output['log_sigma'])
                training_stats.report('Loss/KL', loss_kl)

            loss = loss_recon + loss_kl * self.lambda_kl
            training_stats.report('Loss/loss', loss)

        with torch.autograd.profiler.record_function('main_backward'):
            loss.backward()

        return loss

# ----------------------------------------------------------------------------
