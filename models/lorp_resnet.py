import torch
import torch.nn as nn
from models.lorp_conv import LorpConv2d


class ResBlock(nn.Module):
    def __init__(self, inplane, outplane, use_lorp, rank, bias, conv_type):
        """
        inplane: input channels
        outplane: output channels
        use_lop: whether to use lorp module in this block
        rank: the rank `r` of lorp
        bias: whether to enable bias in lorp
        conv_type: the `conv_type` in lorp.
        """
        super().__init__()
        self.conv_bn_relu_1 = nn.Sequential(
            nn.Conv2d(inplane, outplane, kernel_size=3, padding=1) if not use_lorp
            else LorpConv2d(inplane, outplane, kernel_size=3, padding=1, r=rank, bias=bias, conv_type=conv_type),
            nn.BatchNorm2d(outplane),
            nn.ReLU()
        )

        self.conv_bn_relu_2 = nn.Sequential(
            nn.Conv2d(outplane, outplane, kernel_size=3, padding=1) if not use_lorp
            else LorpConv2d(outplane, outplane, kernel_size=3, padding=1, r=rank, bias=bias, conv_type=conv_type),
            nn.BatchNorm2d(outplane),
            nn.ReLU()
        )

        self.conv_bn_relu_3 = nn.Sequential(
            nn.Conv2d(outplane, outplane, kernel_size=3, padding=1) if not use_lorp
            else LorpConv2d(outplane, outplane, kernel_size=3, padding=1, r=rank, bias=bias, conv_type=conv_type),
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


class LorpRes(nn.Module):
    def __init__(self, rank, conv_type, bias, lorp_res, num_classes=10):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.res1 = ResBlock(inplane=64, outplane=128, use_lorp=lorp_res, bias=bias, conv_type=conv_type, rank=rank)
        self.conv1 = nn.Sequential(
            LorpConv2d(128, 256, kernel_size=3, padding=1, r=rank, bias=bias, conv_type=conv_type),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.res2 = ResBlock(inplane=256, outplane=512, use_lorp=lorp_res, bias=bias, conv_type=conv_type, rank=rank)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(512, num_classes)

    def compute_feature(self, x):
        y = self.pre(x)
        y = self.res1(y)
        y = self.conv1(y)
        y = self.res2(y)
        y = self.head(y)
        return y

    def forward(self, x):
        y = self.pre(x)
        y = self.res1(y)
        y = self.conv1(y)
        y = self.res2(y)
        y = self.head(y)
        y = self.fc(y)
        return y

    def finetune_parameters(self):
        res = []
        for key, param in self.named_parameters():
            if 'fc' in key or 'side_conv' in key:
                res.append(param)
        return res


if __name__ == '__main__':
    model = LorpRes(2, 'A', True, lorp_res=True).cuda()
    x = torch.randn(1, 3, 32, 32).cuda()
    y = model(x)
    optim = torch.optim.Adam(model.finetune_parameters(), lr=1e-2)


