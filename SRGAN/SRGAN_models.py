import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import sampler


class ResnetBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(num_parameters=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)  # Element-wise residual connection

class upsampling_block(nn.Module):
    def __init__(self, in_channels, upscale_factor):
        super(upsampling_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1, stride=1, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channel = 64, num_block = 16):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_channel, kernel_size=9, stride=1, padding=4, bias=False)
        self.Prelu = nn.PReLU(num_parameters=num_channel)
        self.ResBlock = nn.Sequential(*[ResnetBlock(num_channel) for _ in range(num_block)])
        self.conv2 = nn.Conv2d(in_channels=num_channel, out_channels= num_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(num_channel)
        self.upsample_block = nn.Sequential(
            upsampling_block(num_channel, upscale_factor=2), upsampling_block(num_channel, upscale_factor=2)
        )
        self.conv3 = nn.Conv2d(in_channels=num_channel, out_channels=in_channels, kernel_size=9, padding=4, stride=1, bias=False)

    def forward(self, x):
        out1 = self.Prelu(self.conv1(x))
        out = self.ResBlock(out1)
        out = self.BN(self.conv2(out))
        out = out + out1  # Residual skip
        out = self.upsample_block(out)
        out = self.conv3(out)
        return out
    
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_act=True,
        use_bn=True,
        **kwargs,
    ):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)



