import sys

from torchvision import datasets, transforms
from functools import reduce
from torch.utils.data import Dataset
import os
from PIL import Image


class DatasetConstructor:
    support_dataset = ['mnist', 'cifar10', 'fashion_mnist', 'imagenet-tiny']

    def __init__(self, args):
        self.dataset = args.dataset.lower()
        self.path = args.data_path
        self.resize = args.resize
        assert self.dataset in self.support_dataset

    def get_dataset(self, train=True):
        path = self.path if self.path is not None else './data/' + self.dataset
        if self.dataset == 'mnist':
            if self.resize > 0:
                transform = transforms.Compose([
                    transforms.Resize(self.resize),
                    transforms.ToTensor(),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
            return datasets.MNIST(path, train=train, download=True, transform=transform)

        elif self.dataset == 'cifar10':
            if self.resize > 0:
                transform = transforms.Compose([
                    transforms.Resize(self.resize),
                    transforms.ToTensor(),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
            return datasets.CIFAR10(os.path.join(path, 'CIFAR10'), train=train, download=True, transform=transform)

        elif self.dataset == 'fashion_mnist':
            if self.resize > 0:
                transform = transforms.Compose([
                    transforms.Resize(self.resize),
                    transforms.ToTensor(),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
            return datasets.FashionMNIST(path, train=train, download=True, transform=transform)

        elif self.dataset == 'imagenet-tiny':
            if train:
                folder = 'train'
            else:
                folder = 'val'
            transform = []
            new_size = 64 if self.resize < 0 else self.resize
            if train:
                transform.append(transforms.RandomHorizontalFlip())
                transform.append(transforms.RandomResizedCrop(new_size))
            else:
                transform.append(transforms.Resize(new_size))
            transform.append(transforms.ToTensor())
            transform = transforms.Compose(transform)
            return datasets.ImageFolder(root=os.path.join(path, 'tiny-imagenet-200', folder),
                                        transform=transform)


def calculate_mean_std(train_dataset, test_dataset):
    if train_dataset[0][0].shape[0] == 1:
        res = []
        res_std = []
        for i in range(len(train_dataset)):
            sample = train_dataset[i][0]
            res.append(sample.mean())
            res_std.append(sample.std())

        for i in range(len(test_dataset)):
            sample = test_dataset[i][0]
            res.append(sample.mean())
            res_std.append(sample.std())

        return reduce(lambda x, y: x + y, res) / len(res), reduce(lambda x, y: x + y, res_std) / len(res)
