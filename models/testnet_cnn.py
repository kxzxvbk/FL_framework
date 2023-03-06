import torch.nn as nn
import torch


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, use_bn=True):
        super(CNNBlock, self).__init__()
        self.downsample = downsample
        if downsample:
            self.conv1 = BasicConvBlock(in_channels, out_channels, 3, 2, use_bn=use_bn)
            self.downsample = nn.AvgPool2d(3, 2, 1)
        else:
            self.conv1 = BasicConvBlock(in_channels, out_channels, 3, 1, use_bn=use_bn)
            self.downsample = nn.Identity()
        self.conv23 = nn.Sequential(
            BasicConvBlock(out_channels, out_channels, 3, 1, use_bn=use_bn),
            BasicConvBlock(out_channels, out_channels, 3, 1, use_bn=use_bn),
            BasicConvBlock(out_channels, out_channels, 3, 1, use_bn=use_bn),
        )
    
    def forward(self, x):
        print(x.shape)
        x1 = self.conv1(x)
        x2 = self.conv23(x1)
        return x2


class CNNMean(nn.Module):
    def __init__(self, class_number=10):
        super(CNNMean, self).__init__()
        self.channels = [120, 120, 120, 120]
        self.cnn_pre = BasicConvBlock(3, self.channels[0], 7, 2, padding=3)
        self.cnn_blocks = nn.Sequential(
            CNNBlock(self.channels[0], self.channels[1]),
            CNNBlock(self.channels[1], self.channels[2], downsample=True),
            CNNBlock(self.channels[2], self.channels[3]),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels[-1], class_number)

    def forward(self, x):
        x1 = self.cnn_pre(x)
        x2 = self.cnn_blocks(x1)
        o = self.fc(self.pool(x2).flatten(1))
        return o


class CNNNormal(nn.Module):
    def __init__(self, class_number=10):
        super(CNNNormal, self).__init__()
        self.channels = [32, 64, 128, 256]
        self.cnn_pre = BasicConvBlock(3, self.channels[0], 7, 2, padding=3)
        self.cnn_blocks = nn.Sequential(
            CNNBlock(self.channels[0], self.channels[1]),
            CNNBlock(self.channels[1], self.channels[2], downsample=True),
            CNNBlock(self.channels[2], self.channels[3]),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels[-1], class_number)

    def forward(self, x):
        x1 = self.cnn_pre(x)
        x2 = self.cnn_blocks(x1)
        o = self.fc(self.pool(x2).flatten(1))
        return o


class CNNAntiNormal(nn.Module):
    def __init__(self, class_number=10):
        super(CNNAntiNormal, self).__init__()
        self.channels = [256, 128, 64, 32]
        self.cnn_pre = BasicConvBlock(3, self.channels[0], 7, 2, padding=3)
        self.cnn_blocks = nn.Sequential(
            CNNBlock(self.channels[0], self.channels[1]),
            CNNBlock(self.channels[1], self.channels[2], downsample=True),
            CNNBlock(self.channels[2], self.channels[3]),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels[-1], class_number)

    def forward(self, x):
        x1 = self.cnn_pre(x)
        x2 = self.cnn_blocks(x1)
        o = self.fc(self.pool(x2).flatten(1))
        return o


class CNNNoBN(nn.Module):
    def __init__(self, class_number=10):
        super(CNNNoBN, self).__init__()
        self.channels = [256, 128, 64, 32]
        self.cnn_pre = BasicConvBlock(3, self.channels[0], 7, 2, use_bn=False, padding=3)
        self.cnn_blocks = nn.Sequential(
            CNNBlock(self.channels[0], self.channels[1], use_bn=False),
            CNNBlock(self.channels[1], self.channels[2], downsample=True, use_bn=False),
            CNNBlock(self.channels[2], self.channels[3], use_bn=False),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels[-1], class_number)

    def forward(self, x):
        x1 = self.cnn_pre(x)
        x2 = self.cnn_blocks(x1)
        o = self.fc(self.pool(x2).flatten(1))
        return o


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, use_bn=True, use_activation=True):
        super(BasicConvBlock, self).__init__()
        use_bias = not use_bn
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, bias=use_bias, padding=padding)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()
        if use_activation:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


if __name__ == '__main__':
    x = torch.randn(32, 3, 64, 64)
    model = CNNNoBN()
    print(model(x).shape)
