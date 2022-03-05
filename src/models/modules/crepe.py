import functools

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

CENTS_PER_BIN = 20  # cents
PITCH_BINS = 360
SAMPLE_RATE = 16000  # hz
WINDOW_SIZE = 1024  # samples


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    cents = CENTS_PER_BIN * bins + 1997.3794084376191

    # Trade quantization error for noise
    return cents


def cents_to_frequency(cents):
    """Converts cents to frequency in Hz"""
    return 10 * 2 ** (cents / 1200)


def weighted_argmax(logits):
    """Sample observations using weighted sum near the argmax"""
    # Find center of analysis window
    bins = logits.argmax(dim=1)

    # Find bounds of analysis window
    start = torch.max(torch.tensor(0, device=logits.device), bins - 4)
    end = torch.min(torch.tensor(logits.size(1), device=logits.device), bins + 5)

    # Mask out everything outside of window
    for batch in range(logits.size(0)):
        logits[batch, : start[batch]] = 0.0
        logits[batch, end[batch] :] = 0.0

    # Construct weights
    if not hasattr(weighted_argmax, "weights"):
        weights = bins_to_cents(torch.arange(360))
        weighted_argmax.weights = weights[None, :]

    # Ensure devices are the same (no-op if they are)
    weighted_argmax.weights = weighted_argmax.weights.to(logits.device)

    # Apply weights
    cents = (weighted_argmax.weights * logits).sum(dim=1, keepdims=True) / logits.sum(dim=1, keepdims=True)

    # Convert to frequency in Hz
    return cents


###########################################################################
# Model definition
###########################################################################


class Crepe(nn.Module):
    """Crepe model definition"""

    def __init__(self, model="full"):
        super().__init__()

        # Model-specific layer parameters
        if model == "full":
            in_channels = [1, 1024, 128, 128, 128, 256]
            out_channels = [1024, 128, 128, 128, 256, 512]
            self.in_features = 2048
        elif model == "tiny":
            in_channels = [1, 128, 16, 16, 16, 32]
            out_channels = [128, 16, 16, 16, 32, 64]
            self.in_features = 256
        else:
            raise ValueError(f"Model {model} is not supported")

        # Shared layer parameters
        kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]

        # Overload with eps and momentum conversion given by MMdnn  # noqa
        batch_norm_fn = functools.partial(nn.BatchNorm2d, eps=0.0010000000474974513, momentum=0.0)

        # Layer definitions
        self.conv1 = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_sizes[0], stride=strides[0]
        )
        self.conv1_BN = batch_norm_fn(num_features=out_channels[0])

        self.conv2 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_sizes[1], stride=strides[1]
        )
        self.conv2_BN = batch_norm_fn(num_features=out_channels[1])

        self.conv3 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_sizes[2], stride=strides[2]
        )
        self.conv3_BN = batch_norm_fn(num_features=out_channels[2])

        self.conv4 = nn.Conv2d(
            in_channels=in_channels[3], out_channels=out_channels[3], kernel_size=kernel_sizes[3], stride=strides[3]
        )
        self.conv4_BN = batch_norm_fn(num_features=out_channels[3])

        self.conv5 = nn.Conv2d(
            in_channels=in_channels[4], out_channels=out_channels[4], kernel_size=kernel_sizes[4], stride=strides[4]
        )
        self.conv5_BN = batch_norm_fn(num_features=out_channels[4])

        self.conv6 = nn.Conv2d(
            in_channels=in_channels[5], out_channels=out_channels[5], kernel_size=kernel_sizes[5], stride=strides[5]
        )
        self.conv6_BN = batch_norm_fn(num_features=out_channels[5])

        self.classifier = nn.Linear(in_features=self.in_features, out_features=PITCH_BINS)

    def forward(self, x):
        # Forward pass through first five layers
        embed = self.embed(x)

        # Forward pass through layer six
        x = self.layer(embed, self.conv6, self.conv6_BN)

        # shape=(batch, self.in_features)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.in_features)

        # Compute logits
        return torch.sigmoid(self.classifier(x)), embed

    ###########################################################################
    # Forward pass utilities
    ###########################################################################

    def embed(self, x):
        """Map input audio to pitch embedding"""
        # shape=(batch, 1, 1024, 1)
        x = x[:, None, :, None]

        # Forward pass through first five layers
        x = self.layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254))
        x = self.layer(x, self.conv2, self.conv2_BN)
        x = self.layer(x, self.conv3, self.conv3_BN)
        x = self.layer(x, self.conv4, self.conv4_BN)
        x = self.layer(x, self.conv5, self.conv5_BN)

        return x

    @staticmethod
    def layer(x, conv, batch_norm, padding=(0, 0, 31, 32)):
        """Forward pass through one layer"""
        x = F.pad(x, padding)
        x = conv(x)
        x = F.relu(x)
        x = batch_norm(x)
        return F.max_pool2d(x, (2, 1), (2, 1))
