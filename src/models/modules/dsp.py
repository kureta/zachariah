import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F  # noqa
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from .utils import get_frames, pad_audio


class Stretch(nn.Module):
    def __init__(self, n_channels, n_features, size=None, scale_factor=None):
        super().__init__()
        self.stretch = nn.Upsample(size=size, scale_factor=scale_factor, mode="linear", align_corners=True)
        self.pre_stretch = Rearrange("b t c o -> b (c o) t", c=n_channels, o=n_features)
        self.post_stretch = Rearrange("b (c o) t -> b t c o", c=n_channels, o=n_features)

    def forward(self, x):
        x = self.pre_stretch(x)
        x = self.stretch(x)
        x = self.post_stretch(x)

        return x


class HarmonicOscillator(nn.Module):
    def __init__(self, sample_rate: int = 16000, hop_length: int = 64, n_harmonics: int = 64, n_channels: int = 1):
        super().__init__()

        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.hop_size = hop_length
        self.n_channels = n_channels

        harmonics = torch.arange(1, self.n_harmonics + 1, step=1)
        self.register_buffer("harmonics", harmonics, persistent=False)

        self.stretch = Stretch(self.n_channels, self.n_harmonics, scale_factor=self.hop_size)

        self.sum_sinusoids = Reduce("b t c o -> b t c", "sum")

    def forward(self, f0: torch.Tensor, master_amplitude: torch.Tensor, overtone_amplitudes: torch.Tensor):
        # base_pitch.shape = [batch, time, n_channels]
        # amplitude.shape = [batch, time, n_channels]
        # harmonic_distribution = [batch, time, n_channels, n_harmonics]

        # Calculate overtone frequencies
        overtone_fs = torch.einsum("btc,o->btco", f0, self.harmonics)
        # Convert overtone frequencies from Hz to radians / sample
        overtone_fs *= 2 * np.pi
        overtone_fs /= self.sample_rate

        # set amplitudes of overtones above Nyquist to 0.0
        overtone_amplitudes[overtone_fs > np.pi] = 0.0
        # normalize harmonic_distribution so it always sums to one
        # TODO: maybe we should do this before filtering above Nyquist
        overtone_amplitudes /= torch.sum(overtone_amplitudes, dim=3, keepdim=True)
        # scale individual overtone amplitudes by the master amplitude
        overtone_amplitudes = torch.einsum("btco,btc->btco", overtone_amplitudes, master_amplitude)

        # stretch controls by hop_size
        overtone_fs = self.stretch(overtone_fs)
        overtone_amplitudes = self.stretch(overtone_amplitudes)

        # calculate phases and sinusoids
        phases = torch.cumsum(overtone_fs, dim=1)
        sinusoids = torch.sin(phases)

        # scale sinusoids by their corresponding amplitudes and sum them to get the final signal
        sinusoids = torch.einsum("btco,btco->btco", sinusoids, overtone_amplitudes)
        signal = self.sum_sinusoids(sinusoids)

        return signal


class FilteredNoise(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        window_length: int = 1024,
        hop_length: int = 64,
        n_bands: int = 128,
        n_channels: int = 1,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_bands = n_bands
        self.n_channels = n_channels

        self.stretch = nn.Upsample(self.window_length // 2, mode="nearest")

    def forward(self, filter_bands):
        # filter_bands.shape = [batch, time, n_channels, n_bands]

        # Generate white noise
        batch_size, n_steps, _, _ = filter_bands.shape
        # noise.shape = [batch, time, n_channels]
        noise = torch.rand(batch_size, n_steps * self.hop_length, self.n_channels) * 2 - 1

        # Get frames
        padded_noise = pad_audio(noise, self.window_length, self.hop_length)
        # noise_frames.shape = [batch, n_frames (time), n_channels, n_sample (window_length)]
        noise_frames = get_frames(padded_noise, self.window_length, self.hop_length)
        # TODO: should we window noise frames here?

        # Stretch filter to window_length // 2
        filter_ = rearrange(filter_bands, "b t c f -> (b t) c f")
        filter_ = self.stretch(filter_)
        filter_ = rearrange(filter_, "(b t) c f -> b t c f", b=batch_size, t=n_steps)

        # Prepend 0 DC offset
        dc = torch.zeros(*filter_.shape[:-1], 1)
        filter_ = torch.concat([dc, filter_], dim=-1)

        # apply filter to noise
        fft_noise_frames = fft.rfft(noise_frames)
        filtered_fft_noise_frames = filter_ * fft_noise_frames
        filtered_noise_frames = fft.irfft(filtered_fft_noise_frames)
        filtered_noise_frames *= torch.hann_window(self.window_length, periodic=False)

        # overlap add
        # TODO: I forgot what I have done here, but it seems to work
        b, c = filtered_noise_frames.shape[0], filtered_noise_frames.shape[2]
        stacked_noise = rearrange(filtered_noise_frames, "b t c w -> (b c) w t")
        filtered_noise = F.fold(
            stacked_noise, (1, padded_noise.shape[1]), (1, self.window_length), stride=(1, self.hop_length)
        )
        filtered_noise = rearrange(filtered_noise, "(b c) 1 1 t -> b t c", b=b, c=c)
        filtered_noise = filtered_noise[:, self.hop_length :, :]

        return filtered_noise
