from torchvision import datasets, transforms
from functools import reduce
import os
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch


class DatasetConstructor:
    support_dataset = ['mnist', 'cifar10', 'fashion_mnist', 'wiki_text2']

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
            return datasets.CIFAR10(os.path.join(path, 'CIFAR10'), train=train, download=False, transform=transform)

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

        elif self.dataset == 'wiki_text2':
            train_iter = WikiText2(split='train')
            tokenizer = get_tokenizer('basic_english')
            vocab = build_vocab_from_iterator(map(tokenizer, train_iter))
            print(len(vocab))

            def batch_get(voc, cont):
                return [voc[k] for k in cont]

            def data_process(raw_text_iter):
                """Converts raw text into a flat Tensor."""
                data = [torch.tensor(batch_get(vocab, tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
                return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

            def batchify(data, bsz: int):
                """Divides the data into bsz separate sequences, removing extra elements
                that wouldn't cleanly fit.

                Args:
                    data: Tensor, shape [N]
                    bsz: int, batch size

                Returns:
                    Tensor of shape [N // bsz, bsz]
                """
                seq_len = data.size(0) // bsz
                data = data[:seq_len * bsz]
                data = data.view(bsz, seq_len).t().contiguous()
                return data

            # train_iter was "consumed" by the process of building the vocab,
            # so we have to create it again
            train_iter, val_iter, test_iter = WikiText2()
            train_data = data_process(train_iter)
            val_data = data_process(val_iter)
            if train:
                return batchify(train_data, 64)
            else:
                return batchify(val_data, 64)


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


class MyTextLoader:
    def __init__(self, source):
        self.source = source
        self._ind = 0

    def __iter__(self):
        while self._ind < self.source.size(0) - 1:
            yield get_batch(self.source, self._ind)
            self._ind += 1
        self._ind = 0


def get_batch(source, i: int):
    """
            Args:
                source: Tensor, shape [full_seq_len, batch_size]
                i: int

            Returns:
                tuple (data, target), where data has shape [seq_len, batch_size] and
                target has shape [seq_len * batch_size]
    """
    bptt = 64
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


def construct_text_loader(train_data, val_data):
    def batchify(data, bsz: int):
        """Divides the data into bsz separate sequences, removing extra elements
        that wouldn't cleanly fit.

        Args:
            data: Tensor, shape [N]
            bsz: int, batch size

        Returns:
            Tensor of shape [N // bsz, bsz]
        """
        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data

    batch_size = 64
    eval_batch_size = 64
    train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
    val_data = batchify(val_data, eval_batch_size)

    return MyTextLoader(train_data), MyTextLoader(val_data)
