import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from einops import rearrange


class OscillatorBank(nn.Module):
    def __init__(self, n_harmonics: int = 60, sample_rate: int = 16000, hop_length: int = 64):
        super().__init__()

        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.hop_size = hop_length

        self.harmonics = torch.arange(1, self.n_harmonics + 1, step=1)

    def get_harmonic_frequencies(self, f0: torch.Tensor) -> torch.Tensor:
        # f0.shape = [batch, time, ch]
        # Calculate harmonic frequencies in cycles per seconds (Hz).
        harmonic_frequencies = torch.einsum("ijk,k->ijk", f0, self.harmonics)

        harmonic_frequencies *= 2 * np.pi  # radians per second
        harmonic_frequencies /= self.sample_rate  # radians per sample

        # harmonic_frequencies.shape = [batch, time, n_harmonics]
        return harmonic_frequencies

    def get_phases(self, harmonic_frequencies: torch.Tensor) -> torch.Tensor:
        # harmonic_frequencies.shape = [batch, time, n_harmonics]
        harmonic_frequencies = self.rescale(harmonic_frequencies)
        phases = torch.cumsum(harmonic_frequencies, dim=1)
        phases %= 2 * np.pi
        # phases.shape = [batch, time, n_harmonics]
        return phases

    def get_signal(
        self, harmonic_distribution: torch.Tensor, amplitude: torch.Tensor, phases: torch.Tensor
    ) -> torch.Tensor:
        # harmonic_distribution.shape = [batch, time, n_harmonics]
        # amplitude.shape = [batch, time, ch]
        # phases.shape = [batch, time, n_harmonics]
        amplitude = self.rescale(amplitude)
        harmonic_distribution = self.rescale(harmonic_distribution)
        signal = amplitude * harmonic_distribution * torch.sin(phases)
        signal = torch.sum(signal, dim=2)
        # signal.shape = [batch, time, ch]
        return signal

    def rescale(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [batch, time, ch]
        x = F.interpolate(
            rearrange(x, "b t c -> b c 1 t"),
            scale_factor=(1, float(self.hop_size)),
            mode="bicubic",
            align_corners=True,
        )

        # output.shape = [batch, time, ch]
        return rearrange(x, "b c 1 t -> b t c")

    def forward(self, f0: torch.Tensor, amplitude: torch.Tensor, harmonic_distribution: torch.Tensor) -> torch.Tensor:
        # f0.shape = [batch, time, ch]
        # amplitude.shape = [batch_time, ch]
        # harmonic_distribution.shape = [batch, time, n_harmonics]
        harmonic_frequencies = self.get_harmonic_frequencies(f0)
        phases = self.get_phases(harmonic_frequencies)

        # Zero out above nyquist
        mask = harmonic_frequencies > np.pi
        harmonic_distribution = harmonic_distribution.masked_fill(mask, 0.0)

        signal = self.get_signal(harmonic_distribution, amplitude, phases)

        # signal.shape = [batch, time, channel]
        return signal
