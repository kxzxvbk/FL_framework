import torch
import torch.nn as nn


class CNNMean(nn.Module):
    def __init__(self, class_number=10):
        super(CNNMean, self).__init__()
        self.channels = [120, 120, 120, 120]
        self.cnn_pre = BasicConvBlock(3, self.channels[0], 7, 2)
        self.group1 = nn.Sequential(
            BasicConvBlock(self.channels[0], self.channels[1], 3, 1),
            BasicConvBlock(self.channels[1], self.channels[1], 3, 1)
        )
        self.group2 = nn.Sequential(
            BasicConvBlock(self.channels[1], self.channels[2], 3, 2),
            BasicConvBlock(self.channels[2], self.channels[2], 3, 1)
        )
        self.group3 = nn.Sequential(
            BasicConvBlock(self.channels[2], self.channels[3], 3, 1),
            BasicConvBlock(self.channels[3], self.channels[3], 3, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels[-1], class_number)

    def forward(self, x):
        x1 = self.cnn_pre(x)
        x2 = self.group1(x1)
        x3 = self.group2(x2)
        x4 = self.group3(x3)
        o = self.fc(self.pool(x4).flatten(1))
        return o


class CNNNormal(nn.Module):
    def __init__(self, class_number=10):
        super(CNNNormal, self).__init__()
        self.channels = [32, 64, 128, 256]
        self.cnn_pre = BasicConvBlock(3, self.channels[0], 7, 2)
        self.group1 = nn.Sequential(
            BasicConvBlock(self.channels[0], self.channels[1], 3, 1),
            BasicConvBlock(self.channels[1], self.channels[1], 3, 1)
        )
        self.group2 = nn.Sequential(
            BasicConvBlock(self.channels[1], self.channels[2], 3, 2),
            BasicConvBlock(self.channels[2], self.channels[2], 3, 1)
        )
        self.group3 = nn.Sequential(
            BasicConvBlock(self.channels[2], self.channels[3], 3, 1),
            BasicConvBlock(self.channels[3], self.channels[3], 3, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels[-1], class_number)

    def forward(self, x):
        x1 = self.cnn_pre(x)
        x2 = self.group1(x1)
        x3 = self.group2(x2)
        x4 = self.group3(x3)
        o = self.fc(self.pool(x4).flatten(1))
        return o


class CNNAntiNormal(nn.Module):
    def __init__(self, class_number=10):
        super(CNNAntiNormal, self).__init__()
        self.channels = [256, 128, 64, 32]
        self.cnn_pre = BasicConvBlock(3, self.channels[0], 7, 2)
        self.group1 = nn.Sequential(
            BasicConvBlock(self.channels[0], self.channels[1], 3, 1),
            BasicConvBlock(self.channels[1], self.channels[1], 3, 1)
        )
        self.group2 = nn.Sequential(
            BasicConvBlock(self.channels[1], self.channels[2], 3, 2),
            BasicConvBlock(self.channels[2], self.channels[2], 3, 1)
        )
        self.group3 = nn.Sequential(
            BasicConvBlock(self.channels[2], self.channels[3], 3, 1),
            BasicConvBlock(self.channels[3], self.channels[3], 3, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels[-1], class_number)

    def forward(self, x):
        x1 = self.cnn_pre(x)
        x2 = self.group1(x1)
        x3 = self.group2(x2)
        x4 = self.group3(x3)
        o = self.fc(self.pool(x4).flatten(1))
        return o


class CNNNoBN(nn.Module):
    def __init__(self, class_number=10):
        super(CNNNoBN, self).__init__()
        self.channels = [256, 128, 64, 32]
        self.cnn_pre = BasicConvBlock(3, self.channels[0], 7, 2, use_bn=False)
        self.group1 = nn.Sequential(
            BasicConvBlock(self.channels[0], self.channels[1], 3, 1, use_bn=False),
            BasicConvBlock(self.channels[1], self.channels[1], 3, 1, use_bn=False)
        )
        self.group2 = nn.Sequential(
            BasicConvBlock(self.channels[1], self.channels[2], 3, 2, use_bn=False),
            BasicConvBlock(self.channels[2], self.channels[2], 3, 1, use_bn=False)
        )
        self.group3 = nn.Sequential(
            BasicConvBlock(self.channels[2], self.channels[3], 3, 1, use_bn=False),
            BasicConvBlock(self.channels[3], self.channels[3], 3, 1, use_bn=False)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels[-1], class_number)

    def forward(self, x):
        x1 = self.cnn_pre(x)
        x2 = self.group1(x1)
        x3 = self.group2(x2)
        x4 = self.group3(x3)
        o = self.fc(self.pool(x4).flatten(1))
        return o


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_bn=True):
        super(BasicConvBlock, self).__init__()
        use_bias = not use_bn
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                              bias=use_bias)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))
