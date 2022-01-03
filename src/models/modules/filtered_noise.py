import torch
import torch.nn as nn

from ..modules.utils import fft_convolve


class FilteredNoise(nn.Module):
    """Synthesize audio by filtering white noise."""

    def __init__(self, n_samples: int = 64000, window_size: int = 257, initial_bias: float = -5.0):
        """Construct an instance."""
        super().__init__()
        self.n_samples = n_samples
        self.window_size = window_size
        self.initial_bias = initial_bias

    def apply_window_to_impulse_response(self, impulse_response: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """Apply a window to an impulse response and put in causal form.

        Args:
            impulse_response: A series of impulse responses frames to window, of shape
                [batch, n_frames, ir_size].
            causal: Impulse response input is in causal form (peak in the middle).

        Returns:
            impulse_response: Windowed impulse response in causal form, with last
                dimension cropped to window_size if window_size is greater than 0 and less
                than ir_size.
        """
        # impulse_response = torch_float32(impulse_response)

        # If IR is in causal form, put it in zero-phase form.
        if causal:
            impulse_response = torch.fft.fftshift(impulse_response, dim=-1)

        # Get a window for better time/frequency resolution than rectangular.
        # Window defaults to IR size, cannot be bigger.
        ir_size = int(impulse_response.shape[-1])
        if (self.window_size <= 0) or (self.window_size > ir_size):
            window_size = ir_size
        else:
            window_size = self.window
        window = torch.hann_window(window_size, device=impulse_response.device)

        # Zero pad the window and put in zero-phase form.
        padding = ir_size - window_size
        if padding > 0:
            half_idx = (window_size + 1) // 2
            window = torch.cat([window[half_idx:], torch.zeros([padding]), window[:half_idx]], dim=0)
        else:
            window = torch.fft.fftshift(window, dim=-1)

        # Apply the window, to get new IR (both in zero-phase form).
        window = torch.broadcast_to(window, impulse_response.shape)
        impulse_response = window * impulse_response

        # Put IR in causal form and trim zero padding.
        if padding > 0:
            first_half_start = (ir_size - (half_idx - 1)) + 1
            second_half_end = half_idx + 1
            impulse_response = torch.cat(
                [impulse_response[..., first_half_start:], impulse_response[..., :second_half_end]], dim=-1
            )
        else:
            impulse_response = torch.fft.fftshift(impulse_response, dim=-1)

        return impulse_response

    def frequency_impulse_response(self, magnitudes: torch.Tensor) -> torch.Tensor:
        """Get windowed impulse responses using the frequency sampling method.

        Follows the approach in:
        https://ccrma.stanford.edu/~jos/sasp/Windowing_Desired_Impulse_Response.html

        Args:
            magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
                n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
                last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
                f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
                audio into equally sized frames to match frames in magnitudes.

        Returns:
            impulse_response: Time-domain FIR filter of shape
                [batch, frames, window_size] or [batch, window_size].

        Raises:
            ValueError: If window size is larger than fft size.
        """
        # Get the IR (zero-phase form).
        impulse_response = torch.fft.irfft(magnitudes)

        # Window and put in causal form.
        impulse_response = self.apply_window_to_impulse_response(impulse_response)

        return impulse_response

    def frequency_filter(self, audio: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:
        """Filter audio with a finite impulse response filter.

        Args:
            audio: Input audio. Tensor of shape [batch, audio_timesteps].
            magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
                n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
                last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
                f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
                audio into equally sized frames to match frames in magnitudes.

        Returns:
            Filtered audio. Tensor of shape
                [batch, audio_timesteps + window_size - 1] ("valid" padding) or shape
                [batch, audio_timesteps] ("same" padding).
        """
        impulse_response = self.frequency_impulse_response(magnitudes)
        return fft_convolve(audio, impulse_response, padding="same")

    def forward(self, magnitudes):
        """Synthesize audio with filtered white noise.

        Args:
            magnitudes: Magnitudes tensor of shape [batch, n_frames, n_filter_banks].
                Expects float32 that is strictly positive.

        Returns:
            signal: A tensor of harmonic waves of shape [batch, n_samples, 1].
        """
        batch_size = int(magnitudes.shape[0])
        signal = torch.rand([batch_size, self.n_samples], device=magnitudes.device) * 2.0 - 1.0
        return self.frequency_filter(signal, magnitudes)
