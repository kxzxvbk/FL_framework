import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, inplane, outplane):
        super().__init__()
        self.conv_bn_relu_1 = nn.Sequential(
            nn.Conv2d(inplane, outplane, kernel_size=3, padding=1),
            nn.BatchNorm2d(outplane),
            nn.ReLU()
        )

        self.conv_bn_relu_2 = nn.Sequential(
            nn.Conv2d(outplane, outplane, kernel_size=3, padding=1),
            nn.BatchNorm2d(outplane),
            nn.ReLU()
        )

        self.conv_bn_relu_3 = nn.Sequential(
            nn.Conv2d(outplane, outplane, kernel_size=3, padding=1),
            nn.BatchNorm2d(outplane),
            nn.ReLU()
        )

        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv_bn_relu_1(inputs)
        x = self.max_pool2d(x)
        y = self.conv_bn_relu_2(x)
        y = self.conv_bn_relu_3(y)
        x = x + y
        return x


class CifarRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.res1 = ResBlock(inplane=64, outplane=128)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.res2 = ResBlock(inplane=256, outplane=512)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        y = self.pre(x)
        y = self.res1(y)
        y = self.conv1(y)
        y = self.res2(y)
        y = self.head(y)
        return y
