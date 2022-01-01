import numpy as np
import torch
import torch.nn as nn


class MorleyTransform(nn.Module):
    def __init__(self, sample_rate, win_length, n_harmonics, half_bandwidth=1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.n = torch.arange(win_length, dtype=torch.float32)
        self.k = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        self.tp = 1.0 / half_bandwidth

    def generate_morley_matrix(self, f0):
        tp = self.tp * self.sample_rate
        fc = torch.einsum("ijk,k->ijk", f0, self.k) / self.sample_rate
        fc_n = torch.einsum("ijk,l->ijkl", fc, self.n)

        normalizer = (1 / np.sqrt(np.pi * tp)).astype("float32")
        gauss = torch.exp(-((self.n - self.win_length // 2) ** 2) / tp)
        exp = torch.exp(-2j * np.pi * fc_n)
        result = normalizer * gauss * exp

        # Cut above nyquist
        result[fc > 0.5] = 0.0

        return result

    def forward(self, audio_frames, f0):
        morlet = self.generate_morley_matrix(f0)
        transform = torch.einsum("bthn,btn->bth", morlet, audio_frames.type(torch.complex64))
        transform = torch.abs(transform)
        amp = torch.sum(transform, dim=-1, keepdim=True)
        harmonic_distribution = transform / amp
        amp *= 2.3823  # experimentally found normalization factor

        return harmonic_distribution, amp.squeeze(-1)
