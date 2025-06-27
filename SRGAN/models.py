import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import sampler
from torchvision.models import vgg19

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