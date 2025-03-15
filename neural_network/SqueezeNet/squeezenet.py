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
        super(FireModule, self).__init__()
        self.squeeze_channel = nn.Conv2d(in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expansionX1_channels = nn.Conv2d(in_channels=squeeze_channels, out_channels=expansionX1_channels, kernel_size=1)
        self.expansionX1_activation = nn.ReLU(inplace=True)
        self.expansionX3_channels = nn.Conv2d(in_channels=squeeze_channels, out_channels=expansionX3_channels, kernel_size=3, padding=1)
        self.expansionX3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Input
        x : Tensor of shape (batch size, in_channels, height, width)
        Returns:
        Tensor of shape (batch size, expansionX1_channels + expansionX3_channels, height, width)
        """
        # Squeeze layer
        x = self.squeeze_activation(self.squeeze_channel(x))

        # Expansion layers
        expand1 = self.expansionX1_activation(self.expansionX1_channels(x))
        expand3 = self.expansionX3_activation(self.expansionX3_channels(x))

        # Concatenate along the channel dimension (dim=1)
        return torch.cat((expand1, expand3), dim=1)


class SqueezeNet(nn.Module):
    def __init__(self, in_channels, num_classess=1000):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=7, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        #Fire Module
        self.Fire2 = FireModule(96, 16, 64, 64)
        self.Fire3 = FireModule(128, 16, 64, 64)
        self.Fire4 = FireModule(128, 32, 128, 128)
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)

        #Fire Module 
        self.Fire5 = FireModule(256, 32, 128, 128)
        self.Fire6 = FireModule(256, 48, 192, 192)
        self.Fire7 = FireModule(384, 48, 192, 192) 
        self.Fire8 = FireModule(384, 64, 256, 256)

        self.maxpool8 = nn.MaxPool2d(kernel_size=3, stride=2)    

        self.Fire9 = FireModule(512, 64, 256, 256)

        self.dropout = nn.Dropout(p=0.5)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=num_classess, kernel_size=1, stride=1)
        self.avgpoll10 = nn.AdaptiveAvgPool2d((1,1))


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.Fire2(x)
        x = self.Fire3(x)
        x = self.Fire4(x)
        x = self.maxpool4(x)
        x = self.Fire5(x)
        x = self.Fire6(x)
        x = self.Fire7(x)
        x = self.Fire8(x)
        x = self.maxpool8(x)
        x = self.Fire9(x)

        x = self.dropout(x)
        x = self.conv10(x)
        x = self.avgpoll10(x)

        x = torch.flatten(x, 1)

        return x 



