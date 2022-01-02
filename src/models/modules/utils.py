import torch.nn.functional as F  # noqa


def pad_audio(x, win_length, hop_length, strict=True):
    # x.shape = [batch, time, ch]
    # This pads audio so that the middle of the fft window is on the middle of audio frames.
    length = x.shape[1]
    if length % hop_length != 0:
        if strict:
            raise ValueError("In strict mode, audio length must be a multiple of hop length")
        else:
            padding_right = hop_length - length % hop_length
            x = F.pad(x, (0, 0, 0, padding_right))

    padding = (win_length - hop_length) // 2
    x = F.pad(x, (0, 0, padding, padding))

    # return shape = [batch, time]
    return x


def pad_audio_basic(x, win_length, hop_length, strict=True):
    # x.shape = [batch, time, ch]
    # This pads audio so that the middle of the first fft window is on the beginning of the audio.
    length = x.shape[1]
    if length % hop_length != 0:
        if strict:
            raise ValueError("In strict mode, audio length must be a multiple of hop length")
        else:
            padding_right = hop_length - length % hop_length
            x = F.pad(x, (0, 0, 0, padding_right))

    padding_left = win_length // 2
    padding_right = win_length // 2 - hop_length
    x = F.pad(x, (0, 0, padding_left, padding_right))

    # return shape = [batch, time]
    return x


def get_frames(x, win_length, hop_length):
    # x.shape = [batch, time, channel]
    # return shape = [batch, time, channel, win_length]
    return x.unfold(1, win_length, hop_length)
