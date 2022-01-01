from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa


class OscillatorBank(nn.Module):
    def __init__(self, n_harmonics: int = 60, sample_rate: int = 16000, hop_length: int = 256):
        super().__init__()

        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.hop_size = hop_length

        self.harmonics = torch.arange(1, self.n_harmonics + 1, step=1)

    def prepare_harmonics(self, f0: torch.Tensor, harm_amps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cut above Nyquist and normalize
        # Hz (cycles per second)
        harmonics = torch.einsum("ijk,k->ijk", f0, self.harmonics)
        # zero out above nyquist
        mask = harmonics > self.sample_rate // 2
        harm_amps = harm_amps.masked_fill(mask, 0.0)
        # normalize distribution
        # harm_amps /= harm_amps.sum(-1, keepdim=True)
        harmonics *= 2 * np.pi  # radians per second
        harmonics /= self.sample_rate  # radians per sample
        harmonics = self.rescale(harmonics)
        return harmonics, harm_amps

    @staticmethod
    def generate_phases(harmonics: torch.Tensor) -> torch.Tensor:
        phases = torch.cumsum(harmonics, dim=1)
        phases %= 2 * np.pi
        return phases

    def generate_signal(self, harm_amps: torch.Tensor, loudness: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        loudness = self.rescale(loudness)
        harm_amps = self.rescale(harm_amps)
        signal = loudness * harm_amps * torch.sin(phases)
        signal = torch.sum(signal, dim=2)
        return signal

    def rescale(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x.permute(0, 2, 1),
            scale_factor=float(self.hop_size),
            mode="linear",
            align_corners=True,
        ).permute(0, 2, 1)

    def forward(self, f0: torch.Tensor, loudness: torch.Tensor, harm_amps: torch.Tensor) -> torch.Tensor:
        harmonics, harm_amps = self.prepare_harmonics(f0, harm_amps)
        phases = self.generate_phases(harmonics)
        signal = self.generate_signal(harm_amps, loudness, phases)

        return signal
