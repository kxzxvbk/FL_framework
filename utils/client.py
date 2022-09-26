from models.models import ModelConstructor
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import copy

cnt = 0


def get_optimizer(name, lr, momentum, weights):
    if name == 'sgd':
        return optim.SGD(params=weights, momentum=momentum, lr=lr)
    else:
        print('Unrecognized optimizer: ' + name)
        assert False


def get_loss(name):
    if name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif name == 'MSE':
        return nn.MSELoss()
    else:
        print('Unrecognized loss: ' + name)
        assert False


class Client:
    def __init__(self, train_dataset, args, client_id, test_dataset=None):
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.model = ModelConstructor(args).get_model()
        self.device = args.device if args.device >= 0 else 'cpu'
        self.client_id = client_id
        self.fed_keys = []  # only weights in fed_keys will use fed-learning to gather
        if test_dataset:
            self.test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

        self.start_round = args.start_round

    def set_fed_keys(self, keys):
        self.fed_keys = keys

    def update_model(self, dic):
        dic = copy.deepcopy(dic)
        state_dict = self.model.state_dict()
        state_dict.update(dic)

        self.model.load_state_dict(state_dict)

    def train(self, lr, momentum, optimizer, loss, local_eps=1):
        # print('Training for client:' + str(self.client_id))
        self.model.train()
        self.model.to(self.device)

        correct = 0
        total = 0
        tot_loss = 0

        op = get_optimizer(name=optimizer, lr=lr, momentum=momentum, weights=self.model.parameters())
        criterion = get_loss(loss)

        for epoch in range(local_eps):
            for _, (batch_x, batch_y) in enumerate(self.train_dataloader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
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

    def test(self, loss):
        correct = 0
        total = 0
        tot_loss = 0
        self.model.eval()
        self.model.to(self.device)

        criterion = get_loss(loss)

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
