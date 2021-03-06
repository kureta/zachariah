from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange, reduce

from .crepe import Crepe, weighted_argmax


class CrepeFeaturesAndCents(nn.Module):
    def __init__(self, capacity="tiny"):
        super().__init__()
        self.crepe = Crepe(capacity)
        weights = Path(__file__).parent.parent.parent.parent / "data" / "crepe_weights" / f"{capacity}.pth"
        self.crepe.load_state_dict(torch.load(weights))
        self.crepe.eval()
        for param in self.crepe.parameters():
            param.requires_grad = False

    @staticmethod
    def normalize(x):
        # x.shape = [batch, 1024]
        # Mean-center
        x = x - x.mean(dim=1, keepdim=True)

        # Scale
        # Note: during silent frames, this produces very large values. But
        # this seems to be what the network expects.
        x = x / torch.max(torch.tensor(1e-10, device=x.device), x.std(dim=1, keepdim=True))

        return x

    def forward(self, x):
        # x.shape = [batch, 1024]
        x = self.normalize(x)

        logits, features = self.crepe(x)
        cents = weighted_argmax(logits)

        return cents, features


class Loudness(nn.Module):
    def __init__(self, window_length: int = 1024):
        super().__init__()
        self.window_length = window_length

    def forward(self, frames: torch.Tensor):
        # frames.shape = [batch, window_length]

        # calculate rms
        frames = torch.square(frames)
        frames = reduce(frames, "b w -> b", "sum")
        frames = frames / self.window_length
        frames = torch.sqrt(frames)

        return frames
