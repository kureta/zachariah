import numpy as np
import torch
import torch.nn.functional as F  # noqa


def pad_audio(x, win_length, hop_length, strict=True):
    # x.shape = [batch, time, ch]
    # This pads audio so that the middle of the first fft window is on the beginning of the audio.
    # make x channel first
    x = x.transpose(1, 2)
    length = x.shape[2]
    if length % hop_length != 0:
        if strict:
            raise ValueError("In strict mode, audio length must be a multiple of hop length")
        else:
            padding_right = hop_length - length % hop_length
            x = F.pad(x, (0, padding_right))

    padding_left = win_length // 2
    padding_right = win_length // 2 - hop_length
    x = F.pad(x, (padding_left, padding_right))

    # return shape = [batch, time]
    x = x.transpose(1, 2)
    return x


def get_frames(x, win_length, hop_length):
    # x.shape = [batch, time, channel]
    # return shape = [batch, time, channel, win_length]
    return x.unfold(1, win_length, hop_length)


def exp_sigmoid(
    x: torch.Tensor, exponent: float = 10.0, max_value: float = 2.0, threshold: float = 1e-7
) -> torch.Tensor:
    """Exponential Sigmoid point-wise non-linearity.

    Bounds input to [threshold, max_value] with slope given by exponent.

    Args:
        x: Input tensor.
        exponent: In nonlinear regime (away from x=0), the output varies by this
            factor for every change of x by 1.0.
        max_value: Limiting value at x=inf.
        threshold: Limiting value at x=-inf. Stabilizes training when outputs are
            pushed to 0.

    Returns:
        A tensor with point-wise non-linearity applied.
    """
    return max_value * torch.sigmoid(x) ** np.log(exponent) + threshold
