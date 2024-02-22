import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, activation, last=False):
        """
        2D-DeepCNN block, consists of two 2D-convolutional layers with activation, pooling layer and batch normalization

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            kernel: kernel size
            stride: stride size
            activation: string, type of activation function (Sigmoid, ReLU, Tanh)
            last: bool, determines whether it is the last block in the DeepCNN
        """
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2),
            getattr(nn, activation)(inplace=True) if activation == 'ReLU' else getattr(nn, activation)(),
            nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2),
            getattr(nn, activation)(inplace=True) if activation == 'ReLU' else getattr(nn, activation)(),
            nn.AdaptiveAvgPool2d(1) if last else nn.MaxPool2d(kernel_size=3, stride=1, padding=3 // 2),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ClassifierDeepCNN(nn.Module):
    def __init__(self):
        """
        DeepCNN PQD classifier based on the papers from the group, was used in "Open source dataset generator for power
        quality disturbances with deep-learning reference classifiers", and lastly was discussed in "Neural Architecture
        Search (NAS) for Designing Optimal Power Quality Disturbance Classifiers". It is based on the work
        "A novel deep learning method for the classification of power quality disturbances using deep convolutional
        neural network"
        """
        super(ClassifierDeepCNN, self).__init__()

        self.block1 = ConvBlock(in_channels=3, out_channels=32, kernel=3, stride=1, activation='ReLU')
        self.block2 = ConvBlock(in_channels=32, out_channels=64, kernel=3, stride=1, activation='ReLU')
        self.block3 = ConvBlock(in_channels=64, out_channels=128, kernel=3, stride=1, activation='ReLU', last=True)
        self.out = nn.Sequential(nn.Linear(in_features=128, out_features=256),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(in_features=256, out_features=128),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(128),
                                 nn.Linear(in_features=128, out_features=16),
                                 nn.Softmax(dim=1))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class ClassifierBlackBox(nn.Module):
    def __init__(self, channels=3, width=224, height=224):
        """
        Simple classifier made for black box adversarial attacks.
        Args:
            channels: input channels of each signal
            width: number of pixels in the width dimension
            height: number of pixels in the height dimension
        """
        super(ClassifierBlackBox, self).__init__()

        first_layer_size = channels * width * height
        self.block = nn.Sequential(nn.Linear(in_features=first_layer_size, out_features=64),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(in_features=64, out_features=128),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(in_features=128, out_features=256),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool1d(kernel_size=3, stride=1, padding=3 // 2),
                                   nn.BatchNorm1d(256),
                                   nn.Linear(in_features=256, out_features=128),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool1d(kernel_size=3, stride=1, padding=3 // 2),
                                   nn.BatchNorm1d(128),
                                   nn.Linear(in_features=128, out_features=16),
                                   nn.Softmax(dim=1))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.block(x)
        return x
