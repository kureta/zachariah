import numpy as np
import torch
import torch.nn as nn


# TODO: This consumes too much memory. Find a way to reduce this huge morlet matrix
#       to a series of lower dimensional multiplications
class MorletTransform(nn.Module):
    def __init__(self, sample_rate, win_length, n_harmonics, half_bandwidth=1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.n = torch.arange(win_length, dtype=torch.float32)
        self.k = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        self.tp = 1.0 / half_bandwidth

    def generate_morlet_matrix(self, f0):
        # f0.shape = [batch, time, ch]
        tp = self.tp * self.sample_rate
        fc = torch.einsum("btc,k->btck", f0, self.k) / self.sample_rate
        fc_n = torch.einsum("btck,n->btckn", fc, self.n)

        normalizer = (1 / np.sqrt(np.pi * tp)).astype("float32")
        gauss = torch.exp(-((self.n - self.win_length // 2) ** 2) / tp)
        exp = torch.exp(-2j * np.pi * fc_n)
        result = normalizer * gauss * exp

        # Cut above nyquist
        result[fc > 0.5] = 0.0

        # result.shape = [batch, time, ch, n_harmonics, win_length]
        return result

    def forward(self, audio_frames, f0):
        # audio_frames.shape = [batch, time, ch, win_length]
        # f0.shape = [batch, time, ch]
        morlet = self.generate_morlet_matrix(f0)
        transform = torch.einsum("btckn,btcn->btck", morlet, audio_frames.type(torch.complex64))
        transform = torch.abs(transform)
        amp = torch.sum(transform, dim=-1, keepdim=True)
        harmonic_distribution = transform / amp
        amp *= 2.0
        amp = torch.clip(amp, 0.0, 1.0).squeeze(-1)

        # harmonic_distribution.shape = [batch, time, ch, n_harmonics]
        # amp.shape = [batch, time, ch]
        return harmonic_distribution, amp
