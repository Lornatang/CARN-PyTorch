# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math

import torch
from torch import nn

__all__ = [
    "CARN",
]


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
        )
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rcb(x)
        out = torch.add(out, identity)
        out = self.relu(out)

        return out


class _CascadingBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_CascadingBlock, self).__init__()
        self.rb1 = _ResidualBlock(channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(int(channels * 2), channels, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True)
        )

        self.rb2 = _ResidualBlock(channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(channels * 3), channels, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True)
        )

        self.rb3 = _ResidualBlock(channels)
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(channels * 4), channels, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rb1 = self.rb1(x)
        concat1 = torch.cat([rb1, x], 1)
        conv1 = self.conv1(concat1)

        rb2 = self.rb2(conv1)
        concat2 = torch.cat([concat1, rb2], 1)
        conv2 = self.conv2(concat2)

        rb3 = self.rb3(conv2)
        concat3 = torch.cat([concat2, rb3], 1)
        conv3 = self.conv3(concat3)

        return conv3


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample_block(x)

        return out


class CARN(nn.Module):
    def __init__(self, upscale_factor: int) -> None:
        super(CARN, self).__init__()
        self.entry = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))

        self.cb1 = _CascadingBlock(64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True)
        )
        self.cb2 = _CascadingBlock(64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 64, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True)
        )
        self.cb3 = _CascadingBlock(64)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 64, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True)
        )

        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(UpsampleBlock(64, 2))
        elif upscale_factor == 3:
            upsampling.append(UpsampleBlock(64, 3))
        self.upsampling = nn.Sequential(*upsampling)

        self.exit = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))

        self.register_buffer("mean", torch.Tensor([0.4563, 0.4402, 0.4056]).view(1, 3, 1, 1))

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # The images by subtracting the mean RGB value of the DIV2K dataset.
        x = x.sub_(self.mean).mul_(255.)

        out = self.entry(x)

        cb1 = self.cb1(out)
        concat1 = torch.cat([cb1, out], 1)
        conv1 = self.conv1(concat1)
        cb2 = self.cb2(conv1)
        concat2 = torch.cat([concat1, cb2], 1)
        conv2 = self.conv2(concat2)
        cb3 = self.cb3(conv2)
        concat3 = torch.cat([concat2, cb3], 1)
        conv3 = self.conv3(concat3)

        out = self.upsampling(conv3)

        out = self.exit(out)

        out = out.div_(255.).add_(self.mean)
        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
