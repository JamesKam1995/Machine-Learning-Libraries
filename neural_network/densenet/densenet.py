import torch
import torch.nn as nn
from IPython.display import Image
import torchvision
from torchview import draw_graph

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

"""
Densenet consist of dense block which have sets of dense layer

"""

model_parameters = {}
model_parameters['densenet121'] = [6, 12, 24, 16]
model_parameters['densenet169'] = [6, 12, 32, 32]
model_parameters['densenet201'] = [6, 12, 48, 32]
model_parameters['densenet264'] = [6, 12, 64, 48]

#growth rate
k = 32
compression_factor = 0.5 # reduce thenumber of feature-maps at transition layers

class DenseLayer(nn.Module):
    def __init__(self, in_channels):
        """
        Initialize dense later
        1. Batch Normalization
        2. Conv
        3. ReLU

        Input:
        in_channels: Input image in the form of (N, C, H, W)
        """
        super(DenseLayer, self).__init__()
        self.BN1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=4*k, kernel_size=1, stride=1, padding=0, bias=False)

        self.BN2 = nn.BatchNorm2d(num_features=4*k)
        self.conv2 = nn.Conv2d(in_channels=4*k, out_channels=k, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Input:
        x: input image of tensor(N, C, H, W)

        return
        A tensor of shape (N, C + k, H, W)
        """
        #The input pass through BN -> Relu -> Conv

        x_input = x

        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.BN2(x)
        x = self.relu(x)
        x = self.conv2(x)

        #Concat input and layer tensor
        x = torch.cat([x_input, x], 1)

        return x

class DenseBlock(nn.Module):
    """
    A Dense Block is a sequence of densely connected layers, where each layer receives the concatenated outputs of all preceding layers as input.
    This design facilitates feature reuse and efficient gradient flow throughout the network.

    In this implementation, the `DenseBlock` takes an input tensor and processes it through a specified number of `DenseLayer` instances, concatenating their outputs to produce a final tensor with increased channel depth.

    The output tensor maintains the same spatial dimensions (height and width) as the input tensor but has an increased number of channels due to the concatenation of outputs from each dense layer.
    """
    def __init__(self, layer_num, in_channels):
        """
        Input:
        layer_num : total number of dense layer in the dense block
        in_channels: input number of the channels
        """
        super(DenseBlock, self).__init__()
        self.layer_num = layer_num
        self.deep_nn = nn.ModuleList()

        # Add layers
        for num in range(self.layer_num):
            self.deep_nn.add_module(f"DenseLayer_{num}", DenseLayer(in_channels + k*num))

    def forward(self, x):
        """
        Input
        x : a tensor in shape of (N, C, H, W)

        returns:
        x : a tensor in shape of (N, C+k, H, W)
        """
        x_input =x

        for layer in self.deep_nn:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    """
    downsample the feature map resolution using compression factor
    """
    def __init__(self, in_channels, compression_factor):
        super(TransitionLayer, self).__init__()
        self.BN = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels * compression_factor), kernel_size=1, stride=1, padding = 0, bias = False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.BN(x)
        x = self.conv1(x)
        x = self.avgpool(x)
        return x

class DenseNet(nn.Module):
    def __init__(self,densenet_variant,in_channels,num_classes=1000):

        """
        Creating an initial 7x7 convolution followed by 3 DenseBlock and 3 Transition layers. Concluding this with 4th DenseBlock, 7x7 global average pool and FC layer
        for classification
        Args:
            densenet_variant (list) : list containing the total number of layers in a dense block
            in_channels (int) : input number of channels
            num_classes (int) : Total nnumber of output classes

        """

        super(DenseNet,self).__init__()

        # 7x7 conv with s=2 and maxpool
        self.conv1 = nn.Conv2d(in_channels=in_channels ,out_channels=64 ,kernel_size=7 ,stride=2 ,padding=3 ,bias = False)
        self.BN1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


        # adding 3 DenseBlocks and 3 Transition Layers
        self.deep_nn = nn.ModuleList()
        dense_block_inchannels = 64

        for num in range(len(densenet_variant))[:-1]:

            self.deep_nn.add_module( f"DenseBlock_{num+1}" , DenseBlock( densenet_variant[num] , dense_block_inchannels ) )
            dense_block_inchannels  = int(dense_block_inchannels + k*densenet_variant[num])

            self.deep_nn.add_module( f"TransitionLayer_{num+1}" , TransitionLayer( dense_block_inchannels,compression_factor ) )
            dense_block_inchannels = int(dense_block_inchannels*compression_factor)

        # adding the 4th and final DenseBlock
        self.deep_nn.add_module( f"DenseBlock_{num+2}" , DenseBlock( densenet_variant[-1] , dense_block_inchannels ) )
        dense_block_inchannels  = int(dense_block_inchannels + k*densenet_variant[-1])

        self.BN2 = nn.BatchNorm2d(num_features=dense_block_inchannels)

        # Average Pool
        self.average_pool = nn.AdaptiveAvgPool2d(1)

        # fully connected layer
        self.fc1 = nn.Linear(dense_block_inchannels, num_classes)


    def forward(self,x):
        """
        deep_nn is the module_list container which has all the dense blocks and transition blocks
        """
        x = self.relu(self.BN1(self.conv1(x)))
        x = self.maxpool(x)

        for layer in self.deep_nn:
            x = layer(x)

        x = self.relu(self.BN2(x))
        x = self.average_pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        # print(x.shape)
        return x


