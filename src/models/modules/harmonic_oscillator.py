from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from config.default import Config


class OscillatorBank(nn.Module):
    def __init__(self, conf: Config):
        super().__init__()

        self.n_harmonics = conf.n_harmonics
        self.sample_rate = conf.sample_rate
        self.hop_size = conf.hop_length
        self.batch_size = conf.batch_size
        self.live = False

        self.harmonics: torch.Tensor
        self.register_buffer(
            'harmonics',
            torch.arange(1, self.n_harmonics + 1, step=1), persistent=False
        )

        if self.live:
            self.last_phases: torch.Tensor
            self.register_buffer(
                'last_phases',
                # torch.rand(batch_size, n_harmonics) * 2. * np.pi - np.pi, requires_grad=False
                torch.zeros(self.batch_size, self.n_harmonics), persistent=False
            )

    def prepare_harmonics(self, f0: torch.Tensor,
                          harm_amps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cut above Nyquist and normalize
        # Hz (cycles per second)
        harmonics = (
                self.harmonics.unsqueeze(0).unsqueeze(0).repeat(f0.shape[0], f0.shape[1], 1)
                * f0
        )
        # zero out above nyquist
        mask = harmonics > self.sample_rate // 2
        harm_amps = harm_amps.masked_fill(mask, 0.0)
        harm_amps /= harm_amps.sum(-1, keepdim=True)
        harmonics *= 2 * np.pi  # radians per second
        harmonics /= self.sample_rate  # radians per sample
        harmonics = self.rescale(harmonics)
        return harmonics, harm_amps

    @staticmethod
    def generate_phases(harmonics: torch.Tensor) -> torch.Tensor:
        phases = torch.cumsum(harmonics, dim=1)
        phases %= 2 * np.pi
        return phases

    def generate_signal(
            self, harm_amps: torch.Tensor, loudness: torch.Tensor, phases: torch.Tensor
    ) -> torch.Tensor:
        loudness = self.rescale(loudness)
        harm_amps = self.rescale(harm_amps)
        signal = loudness * harm_amps * torch.sin(phases)
        signal = torch.sum(signal, dim=2)
        return signal

    def rescale(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x.permute(0, 2, 1),
            scale_factor=float(self.hop_size),
            mode='linear',
            align_corners=True,
        ).permute(0, 2, 1)

    def forward(self,
                f0: torch.Tensor,
                loudness: torch.Tensor,
                harm_amps: torch.Tensor) -> torch.Tensor:
        harmonics, harm_amps = self.prepare_harmonics(f0, harm_amps)
        if self.live:
            harmonics[:, 0, :] += self.last_phases  # phase offset from last sample
        phases = self.generate_phases(harmonics)
        if self.live:
            self.last_phases[:] = phases[:, -1, :]  # update phase offset
        signal = self.generate_signal(harm_amps, loudness, phases)

        return signal
