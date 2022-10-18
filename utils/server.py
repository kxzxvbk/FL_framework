import copy
from utils.client import get_loss
import torch
from torchvision import transforms, datasets
import os
import torch.nn as nn


class Server:
    def __init__(self, device, test_loader):
        self.attrs = {}

        self.device = device if device >= 0 else 'cpu'
        self.test_loader = test_loader

    def add_attr(self, name, item):
        self.attrs[name] = copy.deepcopy(item)

    def __getitem__(self, item):
        return self.attrs[item]

    def __setitem__(self, key, value):
        self.attrs[key] = value

    def apply_grad(self, grad, lr=1.):
        state_dict = self.attrs['glob_dict']
        for k in grad:
            state_dict[k] = state_dict[k] + lr * grad[k]

    def test(self, model, test_loader=None, train_epoch=3):
        if test_loader:
            old_loader = self.test_loader
            self.test_loader = test_loader
        model = copy.deepcopy(model)

        criterion = nn.CrossEntropyLoss()
        model.init_eval()
        model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), 0.03,
                                    momentum=0.9,
                                    weight_decay=1e-4,)
        # optimizer = torch.optim.Adam(model.parameters(), 0.03, weight_decay=1e-4)

        augmentation = [transforms.ToTensor(), transforms.Normalize(
            mean=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
            std=[0.24703225141799082, 0.24348516474564, 0.26158783926049628])]
        train_dataset = datasets.CIFAR10(os.path.join('./data', 'CIFAR10'), train=True, download=True,
                                         transform=transforms.Compose(augmentation))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=256, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=False)

        for _ in range(train_epoch):
            for _, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                o = model.forward_eval(batch_x)
                lss = criterion(o, batch_y)
                optimizer.zero_grad()
                lss.backward()
                optimizer.step()

        with torch.no_grad():
            # test
            correct = 0
            total = 0
            tot_loss = 0
            model.eval()
            model.to(self.device)

            test_dataset = datasets.CIFAR10(os.path.join('./data', 'CIFAR10'), train=False,
                                            download=True, transform=transforms.Compose(augmentation))
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=256, shuffle=True,
                num_workers=0, pin_memory=True, drop_last=False)

            for _, (batch_x, batch_y) in enumerate(test_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                o = model.forward_eval(batch_x)
                tot_loss += criterion(o, batch_y).item()
                y_pred = o.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(batch_y.data.view_as(y_pred)).long().sum().item()
                total += batch_y.shape[0]

            avg_acc = correct / total
            avg_loss = tot_loss / total

            print('linear protocol: ' + str(avg_acc) + '      ' + str(avg_loss))
        return avg_acc, avg_loss
