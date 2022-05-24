
import math
import numpy as np
from einops import repeat
import torch
from torch import nn
from .layers import *

class StyleGenerator2D(nn.Module):
    def __init__(self, out_res, out_ch, z_dim, ch_mul=1, ch_max=512, skip_conn=True):
        super().__init__()

        self.skip_conn = skip_conn

        # dict key is the resolution, value is the number of channels
        # a trend in both StyleGAN and BigGAN is to use a constant number of channels until 32x32
        self.channels = {
            4: ch_max,
            8: ch_max,
            16: ch_max,
            32: ch_max,
            64: (ch_max // 2 ** 1) * ch_mul,
            128: (ch_max // 2 ** 2) * ch_mul,
            256: (ch_max // 2 ** 3) * ch_mul,
            512: (ch_max // 2 ** 4) * ch_mul,
            1024: (ch_max // 2 ** 5) * ch_mul,
        }

        self.latent_normalization = PixelNorm()
        self.mapping_network = []
        for i in range(3):
            self.mapping_network.append(EqualLinear(in_channel=z_dim, out_channel=z_dim, lr_mul=0.01, activate=True))
        self.mapping_network = nn.Sequential(*self.mapping_network)

        log_size_in = int(math.log(4, 2))  # 4x4
        log_size_out = int(math.log(out_res, 2))

        self.input = ConstantInput(channel=self.channels[4])

        self.conv1 = ModulatedConv2d(
            in_channel=self.channels[4],
            out_channel=self.channels[4],
            kernel_size=3,
            z_dim=z_dim,
            upsample=False,
            activate=True,
        )

        if self.skip_conn:
            self.to_rgb1 = ToRGB(in_channel=self.channels[4], out_channel=out_ch, z_dim=z_dim, upsample=False)
            self.to_rgbs = nn.ModuleList()

        self.convs = nn.ModuleList()

        in_channel = self.channels[4]
        for i in range(log_size_in + 1, log_size_out + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                ModulatedConv2d(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    kernel_size=3,
                    z_dim=z_dim,
                    upsample=True,
                    activate=True,
                )
            )

            self.convs.append(
                ModulatedConv2d(
                    in_channel=out_channel,
                    out_channel=out_channel,
                    kernel_size=3,
                    z_dim=z_dim,
                    upsample=False,
                    activate=True,
                )
            )

            if self.skip_conn:
                self.to_rgbs.append(ToRGB(in_channel=out_channel, out_channel=out_ch, z_dim=z_dim, upsample=True))

            in_channel = out_channel

        # if not accumulating with skip connections we need final layer to map to out_ch channels
        if not self.skip_conn:
            self.out_rgb = ToRGB(in_channel=out_channel, out_channel=out_ch, z_dim=z_dim, upsample=False)
            self.to_rgbs = [None] * (log_size_out - log_size_in)  # dummy for easier control flow

        if self.skip_conn:
            self.n_layers = len(self.convs) + len(self.to_rgbs) + 2
        else:
            self.n_layers = len(self.convs) + 2

    def process_latents(self, z):
        # output should be list with separate latent code for each conditional layer in the model

        if isinstance(z, list):  # latents already in proper format
            pass
        elif z.ndim == 2:  # standard training, shape [B, ch]
            z = self.latent_normalization(z)
            z = self.mapping_network(z)
            z = [z] * self.n_layers
        elif z.ndim == 3:  # latent optimization, shape [B, n_latent_layers, ch]
            n_latents = z.shape[1]
            z = [self.latent_normalization(self.mapping_network(z[:, i])) for i in range(n_latents)]
        return z

    def forward(self, z):
        z = self.process_latents(z)

        out = self.input(z[0])
        B = out.shape[0]
        out = out.view(B, -1, 4, 4)

        out = self.conv1(out, z[0])

        if self.skip_conn:
            skip = self.to_rgb1(out, z[1])
            i = 2
        else:
            i = 1

        for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2], self.to_rgbs):
            out = conv1(out, z[i])
            out = conv2(out, z[i + 1])

            if self.skip_conn:
                skip = to_rgb(out, z[i + 2], skip)
                i += 3
            else:
                i += 2

        if not self.skip_conn:
            skip = self.out_rgb(out, z[i])
        return skip

