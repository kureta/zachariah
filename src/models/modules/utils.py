import numpy as np
import torch
import torch.nn.functional as F  # noqa
from scipy import fftpack


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
    # x = torch_float32(x)
    return max_value * torch.sigmoid(x) ** np.log(exponent) + threshold


def overlap_and_add(ys: torch.Tensor, hop_length: int) -> torch.Tensor:
    """Overlap add overlapping windows."""
    # TODO: Make sure this works correctly. Write a test.
    return (
        torch.nn.functional.fold(
            ys.transpose(1, 2),
            (1, (ys.shape[1] - 1) * hop_length + ys.shape[2]),
            (1, ys.shape[2]),
            stride=(1, hop_length),
        )
        .squeeze(1)
        .squeeze(1)
    )


def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
    """Calculate final size for efficient FFT.

    Args:
        frame_size: Size of the audio frame.
        ir_size: Size of the convolving impulse response.
        power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
            numbers. TPU requires power of 2, while GPU is more flexible.

    Returns:
        fft_size: Size for efficient FFT.
    """
    convolved_frame_size = ir_size + frame_size - 1
    if power_of_2:
        # Next power of 2.
        fft_size = int(2 ** np.ceil(np.log2(convolved_frame_size)))
    else:
        fft_size = int(fftpack.helper.next_fast_len(convolved_frame_size))
    return fft_size


def crop_and_compensate_delay(
    audio: torch.Tensor, audio_size: int, ir_size: int, padding: str, delay_compensation: int
) -> torch.Tensor:
    """Crop audio output from convolution to compensate for group delay.

    Args:
        audio: Audio after convolution. Tensor of shape [batch, time_steps].
        audio_size: Initial size of the audio before convolution.
        ir_size: Size of the convolving impulse response.
        padding: Either "valid" or "same". For "same" the final output to be the
            same size as the input audio (audio_timesteps). For "valid" the audio is
            extended to include the tail of the impulse response (audio_timesteps +
            ir_timesteps - 1).
        delay_compensation: Samples to crop from start of output audio to compensate
            for group delay of the impulse response. If delay_compensation < 0 it
            defaults to automatically calculating a constant group delay of the
            windowed linear phase filter from frequency_impulse_response().

    Returns:
        Tensor of cropped and shifted audio.

    Raises:
        ValueError: If padding is not either "valid" or "same".
    """
    # Crop the output.
    if padding == "valid":
        crop_size = ir_size + audio_size - 1
    elif padding == "same":
        crop_size = audio_size
    else:
        raise ValueError(f'Padding must be "valid" or "same", instead of {padding}.')

    # Compensate for the group delay of the filter by trimming the front.
    # For an impulse response produced by frequency_impulse_response(),
    # the group delay is constant because the filter is linear phase.
    total_size = int(audio.shape[-1])
    crop = total_size - crop_size
    start = (ir_size - 1) // 2 - 1 if delay_compensation < 0 else delay_compensation
    end = crop - start
    return audio[:, start:-end]


def fft_convolve(
    audio: torch.Tensor, impulse_response: torch.Tensor, padding: str = "same", delay_compensation: int = -1
) -> torch.Tensor:
    """Filter audio with frames of time-varying impulse responses.

    Time-varying filter. Given audio [batch, n_samples], and a series of impulse
    responses [batch, n_frames, n_impulse_response], splits the audio into frames,
    applies filters, and then overlap-and-adds audio back together.
    Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
    convolution for large impulse response sizes.

    Args:
      audio: Input audio. Tensor of shape [batch, audio_timesteps].
      impulse_response: Finite impulse response to convolve. Can either be a 2-D
        Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
        ir_frames, ir_size]. A 2-D tensor will apply a single linear
        time-invariant filter to the audio. A 3-D Tensor will apply a linear
        time-varying filter. Automatically chops the audio into equally shaped
        blocks to match ir_frames.
      padding: Either "valid" or "same". For "same" the final output to be the
        same size as the input audio (audio_timesteps). For "valid" the audio is
        extended to include the tail of the impulse response (audio_timesteps +
        ir_timesteps - 1).
      delay_compensation: Samples to crop from start of output audio to compensate
        for group delay of the impulse response. If delay_compensation is less
        than 0 it defaults to automatically calculating a constant group delay of
        the windowed linear phase filter from frequency_impulse_response().

    Returns:
      audio_out: Convolved audio. Tensor of shape
          [batch, audio_timesteps + ir_timesteps - 1] ("valid" padding) or shape
          [batch, audio_timesteps] ("same" padding).

    Raises:
      ValueError: If audio and impulse response have different batch size.
      ValueError: If audio cannot be split into evenly spaced frames. (i.e. the
        number of impulse response frames is on the order of the audio size and
        not a multiple of the audio size.)
    """
    # audio, impulse_response = torch_float32(audio), torch_float32(impulse_response)

    # Get shapes of audio.
    batch_size, audio_size = audio.shape

    # Add a frame dimension to impulse response if it doesn"t have one.
    ir_shape = impulse_response.shape
    if len(ir_shape) == 2:
        impulse_response = impulse_response[:, None, :]

    # Broadcast impulse response.
    if ir_shape[0] == 1 and batch_size > 1:
        impulse_response = torch.tile(impulse_response, [batch_size, 1, 1])

    # Get shapes of impulse response.
    ir_shape = impulse_response.shape
    batch_size_ir, n_ir_frames, ir_size = ir_shape

    # Validate that batch sizes match.
    if batch_size != batch_size_ir:
        raise ValueError(
            f"Batch size of audio ({batch_size}) and impulse " f"response ({batch_size_ir}) must be the same."
        )

    # Cut audio into frames.
    frame_size = int(np.ceil(audio_size / n_ir_frames))
    hop_size = frame_size
    audio_frames = get_frames(audio, frame_size, hop_size)

    # Check that number of frames match.
    n_audio_frames = int(audio_frames.shape[1])
    if n_audio_frames != n_ir_frames:
        raise ValueError(
            f"Number of Audio frames ({n_audio_frames}) and impulse response frames "
            f"({n_ir_frames}) do not match. For small hop size = ceil(audio_size / n_ir_frames), "
            "number of impulse response frames must be a multiple of the audio size."
        )

    # Pad and FFT the audio and impulse responses.
    fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)
    audio_fft = torch.fft.rfft(audio_frames, fft_size)
    ir_fft = torch.fft.rfft(impulse_response, fft_size)

    # Multiply the FFTs (same as convolution in time).
    audio_ir_fft = torch.multiply(audio_fft, ir_fft)

    # Take the iFFT to re-synthesize audio.
    audio_frames_out = torch.fft.irfft(audio_ir_fft)
    audio_out = overlap_and_add(audio_frames_out, hop_size)

    # Crop and shift the output audio.
    return crop_and_compensate_delay(audio_out, audio_size, ir_size, padding, delay_compensation)
