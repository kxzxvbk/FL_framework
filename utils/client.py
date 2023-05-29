import torch

from models.models import ModelConstructor
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import copy
import random

cnt = 0


def get_optimizer(name, lr, momentum, weights):
    if name.lower() == 'sgd':
        return optim.SGD(params=weights, momentum=momentum, lr=lr)
    elif name.lower() == 'adam':
        return optim.Adam(params=weights, lr=lr)
    else:
        print('Unrecognized optimizer: ' + name)
        assert False


def get_loss(name):
    if name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif name == 'MSE':
        return nn.MSELoss()
    elif name == 'gpt':
        return lambda x, y: x[1]
    else:
        print('Unrecognized loss: ' + name)
        assert False


class Client:
    def __init__(self, train_dataset, args, client_id, test_dataset=None, test_frac=0):
        self.args = args
        if test_frac == 0:
            self.sample_num = len(train_dataset)
            self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        else:
            real_train = copy.deepcopy(train_dataset)
            real_test = copy.deepcopy(train_dataset)
            indexes = real_train.indexes
            random.shuffle(indexes)
            train_index = indexes[:int((1 - test_frac) * len(train_dataset))]
            test_index = indexes[int((1 - test_frac) * len(train_dataset)):]
            real_train.indexes = train_index
            real_test.indexes = test_index
            self.sample_num = len(real_train)

            self.train_dataloader = DataLoader(real_train, batch_size=args.batch_size, shuffle=True)
            self.test_dataloader = DataLoader(real_test, batch_size=args.batch_size, shuffle=True)

        self.model = ModelConstructor(args).get_model()
        self.device = args.device if args.device >= 0 else 'cpu'
        self.client_id = client_id
        self.fed_keys = []  # only weights in fed_keys will use fed-learning to gather

        self.start_round = args.start_round

    def set_fed_keys(self, keys):
        self.fed_keys = keys

    def update_model(self, dic):
        dic = copy.deepcopy(dic)
        state_dict = self.model.state_dict()
        state_dict.update(dic)

        self.model.load_state_dict(state_dict)

    def train(self, lr, momentum, optimizer, loss, local_eps=1, finetune=False, freeze_side=True):
        # Local training.
        self.model.train()
        self.model.to(self.device)

        correct = 0
        total = 0
        tot_loss = 0

        if finetune:
            weights = self.model.finetune_parameters()
        elif freeze_side:
            try:
                weights = self.model.beside_side_parameters()
            except AttributeError:
                print('Model does not support beside_side_parameters. Skipping ... ')
                weights = self.model.parameters()
        else:
            weights = self.model.parameters()
        op = get_optimizer(name=optimizer, lr=lr, momentum=momentum, weights=weights)
        criterion = get_loss(loss)

        for epoch in range(local_eps):
            for _, (batch_x, batch_y) in enumerate(self.train_dataloader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                if 'dataloader_type' in self.args.__dict__.keys() and self.args.dataloader_type == 'nlp':
                    o = self.model(batch_x, batch_y)
                    loss = criterion(o, batch_y)
                    tot_loss += loss.item()
                    _, y_pred = o[0][0].data.max(1, keepdim=True)
                    correct += 1
                    total += batch_y.shape[0]
                else:
                    o = self.model(batch_x)
                    loss = criterion(o, batch_y)
                    tot_loss += loss.item()
                    _, y_pred = o.data.max(1, keepdim=True)
                    correct += y_pred.eq(batch_y.data.view_as(y_pred)).long().sum().item()
                total += batch_y.shape[0]

                op.zero_grad()
                loss.backward()
                op.step()

        avg_acc = correct / total
        avg_loss = tot_loss / total

        self.model.to('cpu')
        return avg_acc, avg_loss

    def finetune(self, lr, momentum, optimizer, loss, local_eps=1):
        # Local training.
        self.model.train()
        self.model.to(self.device)

        # For calculating train loss and train acc.
        correct = 0
        total = 0
        tot_loss = 0
        tot_acces = []
        tot_losses = []

        # Get weights to be finetuned.
        if self.args.finetune_type != 'all':
            weights = self.model.finetune_parameters()
        else:
            weights = self.model.parameters()
        # Get optimizer and loss.
        op = get_optimizer(name=optimizer, lr=lr, momentum=momentum, weights=weights)
        criterion = get_loss(loss)

        # Main loop.
        for epoch in range(local_eps):
            self.model.train()
            self.model.to(self.device)
            for _, (batch_x, batch_y) in enumerate(self.train_dataloader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                # If the task is nlp.
                if 'dataloader_type' in self.args.__dict__.keys() and self.args.dataloader_type == 'nlp':
                    o = self.model(batch_x, batch_y)
                    loss = criterion(o, batch_y)
                    tot_loss += loss.item()
                    _, y_pred = o[0][0].data.max(1, keepdim=True)
                    correct += 1
                    total += batch_y.shape[0]
                # CV task.
                else:
                    o = self.model(batch_x)
                    loss = criterion(o, batch_y)
                    tot_loss += loss.item()
                    _, y_pred = o.data.max(1, keepdim=True)
                    correct += y_pred.eq(batch_y.data.view_as(y_pred)).long().sum().item()
                total += batch_y.shape[0]
                op.zero_grad()
                loss.backward()
                op.step()
            # Test model every epoch.
            acc, loss = self.test('CrossEntropyLoss')
            tot_acces.append(acc)
            tot_losses.append(loss)

        avg_acc = correct / total
        avg_loss = tot_loss / total
        self.model.to('cpu')
        return avg_acc, avg_loss, tot_acces, tot_losses

    def test(self, loss):
        # Test model.
        correct = 0
        total = 0
        tot_loss = 0
        self.model.eval()
        self.model.to(self.device)

        criterion = get_loss(loss)
        with torch.no_grad():
            for _, (batch_x, batch_y) in enumerate(self.test_dataloader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                o = self.model(batch_x)
                tot_loss += criterion(o, batch_y).item()
                y_pred = o.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(batch_y.data.view_as(y_pred)).long().sum().item()
                total += batch_y.shape[0]
        self.model.to('cpu')

        avg_acc = correct / total
        avg_loss = tot_loss / total

        return avg_acc, avg_loss

    def get_state_dict(self, keys):
        state_dict = self.model.state_dict()
        return {k: state_dict[k] for k in keys}
