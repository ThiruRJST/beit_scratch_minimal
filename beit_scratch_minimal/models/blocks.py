"""
What we need?
1. Residual block
2. Build encoder block with residual block
3. Build decoder block with residual block
4. The encoder block must have a embedding table that acts a codebook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


from functools import partial


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):

        residual = x  # storing x for residual

        # First Sub-group
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        # Second Sub-group
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.activation(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(
        self, in_channels: int = 3, out_channels: int = 512, n_layers: int = 3, n_res_blocks: int = 1
    ):

        super().__init__()

        self.layers_list = nn.ModuleList()

        self.hidden_channels = out_channels // 4
        self.layers_list.append(
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=1)
        )

        for _ in range(n_layers):
            self.layers_list.append(
                ResidualBlock(
                    in_channels=self.hidden_channels, out_channels=self.hidden_channels
                )
            )

            self.layers_list.extend(
                [
                    ResidualBlock(
                    in_channels=self.hidden_channels, out_channels=self.hidden_channels
                )
                for _ in range(n_res_blocks)
                ]
            )
            self.layers_list.append(
                nn.Conv2d(
                    self.hidden_channels,
                    self.hidden_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            self.layers_list.append(nn.ReLU())

        self.layers_list.append(
            nn.Conv2d(self.hidden_channels, out_channels, kernel_size=1)
        )

        self.encoder_layers = nn.Sequential(*self.layers_list)

    def forward(self, x):
        return self.encoder_layers(x)


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int = 512, out_channels: int = 3, n_layers:int = 3, n_res_blocks: int = 1
    ):

        super().__init__()

        self.layers_list = nn.ModuleList()

        self.hidden_channels = in_channels // 4
        self.layers_list.append(
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=1)
        )

        for _ in range(n_layers):
            self.layers_list.extend(
                [
                    ResidualBlock(
                    in_channels=self.hidden_channels, out_channels=self.hidden_channels
                )
                for _ in range(n_res_blocks)
                ]
            )

            self.layers_list.append(
                nn.ConvTranspose2d(
                    self.hidden_channels,
                    self.hidden_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            self.layers_list.append(nn.ReLU())

        self.layers_list.append(
            nn.Conv2d(self.hidden_channels, out_channels, kernel_size=1)
        )

        self.decoder_layers = nn.Sequential(*self.layers_list)

    def forward(self, x):
        return self.decoder_layers(x)