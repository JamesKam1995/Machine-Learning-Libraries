import torch
import torch.nn as nn
import torch.nn.functional as F


class FireModule(nn.Module):
    """
    Implement the fire module of SquuezeNet
    consists of 
    1. expansion Layer 
    2. Squeeze Layer
    """
    def __init__(self, in_channels, squeeze_channels, expansionX1_channels, expansionX3_channels):
        super(FireModule).__init__()
        self.in_channels = in_channels
        self.squeeze_channel = nn.Conv2d(in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expansionX1_channels = nn.Conv2d(in_channels=squeeze_channels, out_channels=expansionX1_channels, kernel_size=1)
        self.expansionX1_activation = nn.ReLU(inplace=True)
        self.expansionX3_channels = nn.Conv2d(in_channels=squeeze_channels, out_channels=expansionX3_channels, kernel_size=1, padding=1)
        self.expansionX3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Input
        x : Tensor of shape (batch size, in_channels, height, width)
        Returns:
        Tensor of shape (batch size, expansionX1_channels + expansionX3_channels, height, width)
        """

        x = self.squeeze_activation(self.squeeze_channel(x))
        return torch.cat(
            self.expansionX1_activation(self.expansionX1_channels(x)),
            self.expansionX3_activation(self.expansionX3_channels(x)), 1)


class SqueezeNet(nn.Module):
    def __init__(self, in_channels, num_classess=1000):
        super(SqueezeNet, self).__init__()
        #conv
        self.conv1 = nn.Conv2d(in_channels, out_channels=?, stride=1, padding=0) #check 
        #MaxPool
        #fire module
        """
        1. Fire (96, )
        2. Fire(128, )
        3. Fire (256, )
        Maxpool 
        4. Fire()
        """
        pass

    def forward():
        pass


