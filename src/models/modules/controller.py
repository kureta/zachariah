from collections import OrderedDict
from typing import Tuple, Type

import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    def __init__(self, n_input: int, n_units: int, n_layer: int, relu: Type[nn.Module] = nn.LeakyReLU):
        super().__init__()
        self.n_layer = n_layer
        self.n_input = n_input
        self.n_units = n_units

        layers = [
            [
                nn.Linear(n_input, n_units),
                nn.LayerNorm(normalized_shape=n_units),
                relu(),
            ]
        ]

        for _ in range(1, n_layer):
            layers.append(
                [
                    nn.Linear(n_units, n_units),
                    nn.LayerNorm(normalized_shape=n_units),
                    relu(),
                ]
            )

        mlps = [nn.Sequential(*block) for block in layers]
        self.net = nn.Sequential(OrderedDict(zip((f"mlp_layer{i}" for i in range(1, n_layer + 1)), mlps)))

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x


class Controller(nn.Module):
    def __init__(
        self,
        n_harmonics: int = 120,
        n_noise_filters: int = 100,
        decoder_mlp_units: int = 512,
        decoder_mlp_layers: int = 3,
        decoder_gru_units: int = 512,
        decoder_gru_layers: int = 1,
        batch_size: int = 1,
        live: bool = False,
    ):
        super().__init__()

        self.live = live
        self.batch_size = batch_size

        self.mlp_f0 = MLP(n_input=1, n_units=decoder_mlp_units, n_layer=decoder_mlp_layers)
        self.mlp_loudness = MLP(n_input=1, n_units=decoder_mlp_units, n_layer=decoder_mlp_layers)

        self.num_mlp = 2

        self.gru = nn.GRU(
            input_size=self.num_mlp * decoder_mlp_units,
            hidden_size=decoder_gru_units,
            num_layers=decoder_gru_layers,
            batch_first=True,
        )

        if self.live:
            self.hidden: torch.Tensor
            self.register_buffer(
                "hidden",
                torch.randn(self.gru.num_layers, self.batch_size, self.gru.hidden_size),
            )

        self.mlp_gru = MLP(
            n_input=decoder_gru_units + self.num_mlp * decoder_mlp_units,
            n_units=decoder_mlp_units,
            n_layer=decoder_mlp_layers,
        )

        # one element for overall loudness
        self.dense_harmonic = nn.Linear(decoder_mlp_units, n_harmonics)
        self.dense_loudness = nn.Linear(decoder_mlp_units + 1, 1)
        self.dense_filter = nn.Linear(decoder_mlp_units, n_noise_filters)

    def forward(self, f0: Tensor, loudness: Tensor) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]:
        latent_f0 = self.mlp_f0(f0)
        latent_loudness = self.mlp_loudness(loudness)

        latent = torch.cat((latent_f0, latent_loudness), dim=-1)

        if self.live:
            # self.hidden_.to(f0.device)
            latent, self.hidden[:] = self.gru(latent, self.hidden)
        else:
            latent, _ = self.gru(latent)

        latent = torch.cat((latent, latent_f0, latent_loudness), dim=-1)
        latent = self.mlp_gru(latent)

        harm_amps = self.modified_sigmoid(self.dense_harmonic(latent))
        # TODO: original implementation did not use loudness here.
        total_harm_amp = self.modified_sigmoid(self.dense_loudness(torch.cat((latent, loudness), dim=-1)))

        noise_distribution = self.dense_filter(latent)
        noise_distribution = self.modified_sigmoid(noise_distribution)

        return (f0, total_harm_amp, harm_amps), noise_distribution

    @staticmethod
    def modified_sigmoid(a: Tensor) -> Tensor:
        a = a.sigmoid()
        a = a.pow(2.3026)  # log10
        a = a.mul(2.0)
        a.add_(1e-7)
        return a
