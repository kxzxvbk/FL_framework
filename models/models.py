import torch
import torch.nn as nn
import torchvision


class ModelConstructor:
    """
    neural networks are constructed by this class
    """
    support_models = ['CNNModel', 'MLPModel']

    def __init__(self, args):
        self.args = args

    def get_model(self):
        if self.args.model == 'cnn':
            return CNNModel(class_number=self.args.class_number, input_channel=self.args.input_channel)
        elif self.args.model == 'mlp':
            return MLPModel(input_dim=self.args.input_units,
                            class_number=self.args.class_number, hidden_units=self.args.hidden_unit)
        elif self.args.model == 'lenet':
            return torch.nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2),
                                       nn.Sigmoid(),
                                       nn.AvgPool2d(kernel_size=2, stride=2),
                                       nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                                       nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                                       nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                                       nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 10))
        elif self.args.model == 'alexnet':
            return nn.Sequential(
                nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
                nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                nn.Linear(4096, 10))
        elif self.args.model == 'resnet18':
            net = torchvision.models.resnet18()
            net.fc = nn.Linear(512, self.args.class_number)
            return net
        elif self.args.model == 'resnet50':
            return torchvision.models.resnet50()
        elif self.args.model == 'resnet34':
            return torchvision.models.resnet34()
        else:
            print('Unrecognized model name: ' + self.args.model)


class CNNModel(nn.Module):
    def __init__(self, class_number, input_channel=3):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc = nn.Linear(180, class_number)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc(x)
        return x


class MLPModel(nn.Module):
    def __init__(self, input_dim, class_number, hidden_units=1024):
        super(MLPModel, self).__init__()
        self.layer_input = nn.Linear(input_dim, hidden_units)
        self.layer_hidden = nn.Linear(hidden_units, class_number)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layer_input(x)
        x = torch.dropout(x, 0.5, train=self.training)
        x = torch.relu(x)
        x = self.layer_hidden(x)
        return x
