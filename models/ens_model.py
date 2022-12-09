import torch.nn as nn
import torch


class EnsConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, client_id, client_num):
        super().__init__()
        assert out_channel % client_num == 0
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channel, out_channel // client_num, kernel_size=kernel_size, padding=padding)
            for _ in range(client_num)])
        self.client_id = client_id
        self.client_num = client_num
        self._freeze()

    def forward(self, x):
        return torch.cat([conv(x) for conv in self.convs], dim=1)

    def _freeze(self):
        for i, conv in enumerate(self.convs):
            if i != self.client_id:
                for key, param in conv.named_parameters():
                    param.requires_grad = False


class EnsResBlock(nn.Module):
    def __init__(self, inplane, outplane, client_id, client_num):
        super().__init__()
        self.conv_bn_relu_1 = nn.Sequential(
            EnsConv2d(inplane, outplane, kernel_size=3, padding=1, client_id=client_id, client_num=client_num),
            nn.BatchNorm2d(outplane),
            nn.ReLU()
        )

        self.conv_bn_relu_2 = nn.Sequential(
            EnsConv2d(outplane, outplane, kernel_size=3, padding=1, client_id=client_id, client_num=client_num),
            nn.BatchNorm2d(outplane),
            nn.ReLU()
        )

        self.conv_bn_relu_3 = nn.Sequential(
            EnsConv2d(outplane, outplane, kernel_size=3, padding=1, client_id=client_id, client_num=client_num),
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
    def __init__(self, client_id, client_num, num_classes=10):
        super().__init__()
        self.client_id = client_id
        self.client_num = client_num

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

        self.res2 = EnsResBlock(inplane=256, outplane=512, client_id=client_id, client_num=client_num)
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


if __name__ == '__main__':
    model = CifarRes(1, 4, 10)
    print(model)
    x = torch.randn((32, 3, 32, 32))
    y = model(x)
    print(y.shape)

