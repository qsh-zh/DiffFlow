import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

__all__ = ["FourierMLP"]


class FourierMLP(nn.Module):
    def __init__(self, data_shape, num_layers=2, channels=128):
        super().__init__()
        self.data_shape = [data_shape]

        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channels)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channels)[None])
        self.input_embed = nn.Linear(int(np.prod(data_shape)), channels)
        self.timestep_embed = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.layers = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(channels, int(np.prod(data_shape))),
        )

    def forward(self, inputs, cond):
        sin_embed_cond = torch.sin(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        cos_embed_cond = torch.cos(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        embed_cond = self.timestep_embed(
            rearrange([sin_embed_cond, cos_embed_cond], "d b w -> b (d w)")
        )
        embed_ins = self.input_embed(inputs.view(inputs.shape[0], -1))
        out = self.layers(embed_ins + embed_cond)
        return out.view(inputs.shape)
