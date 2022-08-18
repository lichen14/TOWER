import torch.nn as nn
from .modules import Flatten, Activation
import torch

class SegmentationHead_my(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv2d_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__()
        self.conv2d_3x3 = conv2d_3x3
        self.conv2d_1x1 = conv2d_1x1
        self.upsampling = upsampling
        self.activation = activation

    def forward(self, x):
        # See note [TorchScript super()]
        x_visual = self.conv2d_3x3(x)
        x = self.conv2d_1x1(x_visual)
        x = self.upsampling(x)
        x = self.activation(x)
        return x#,x_visual

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)

class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)

        
class ProjectionHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, pooling="avg"):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        # dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear1 = nn.Linear(in_channels, in_channels, bias=True)
        relu = nn.ReLU(inplace=True)
        linear2 = nn.Linear(in_channels, out_channels, bias=True)
        # activation = Activation(activation)
        super().__init__(pool, flatten, linear1, relu, linear2)

class Projector_0504(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        linear1 = nn.Linear(in_channels, in_channels, bias=True)
        relu = nn.ReLU(inplace=True)
        linear2 = nn.Linear(in_channels, out_channels, bias=True)
        # activation = Activation(activation)
        super().__init__(linear1, relu, linear2)