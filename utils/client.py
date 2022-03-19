from models.models import ModelConstructor
from torch.utils.data import DataLoader
import torch.optim as optim
from functools import reduce
import torch.nn as nn
import torch
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
        self.compress_ratio = args.compress_ratio
        self.increment_svd = args.increment_svd
        self.comp_bias = args.compress_bias
        self.comp_other = args.compress_other
        self.start_round = args.start_round

        # only used in powersgd
        self.qs = {}

        # used in fed_comp
        self.us = {}
        self.es = {}

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

    def setup_powersgd(self, compress_ratio):
        state_dict = self.model.state_dict()

        for key in self.fed_keys:
            if (('conv' in key or 'fc' in key) and 'weight' in key) or 'downsample.0' in key:
                old_shape = state_dict[key].shape[1:]
                flat_dim = reduce(lambda x, y: x * y, old_shape)
                rank = max(flat_dim // compress_ratio, 1)
                q = torch.randn((flat_dim, rank))
                self.qs[key] = q

    def set_u_dict(self, us):
        self.us.update(us)

    def compress(self, glob, key, error_corr):
        global cnt
        if not error_corr:
            grad = (self.model.state_dict()[key] - glob).flatten()
            u = self.us[key]
            comp_grad = u.T @ grad
            decomp_grad = u @ comp_grad
            ratio = (torch.std(grad) + 1e-8) / (torch.std(decomp_grad).item() + 1e-8)

            # if ratio <= 10:
            #     ratio = 2
            if ratio >= 3:
                ratio = 3
            print('ratio of ' + key + '  grad norm: ' + str(torch.std(grad)) + '  decomp norm: ' + str(torch.std(decomp_grad).item()) + '  ratio: '+str(ratio))

            return ratio * comp_grad, torch.zeros_like(grad).float()
        else:
            grad = (self.model.state_dict()[key] - glob).flatten()
            u = self.us[key]
            # y = u.T @ grad @ (u.T @ u).inverse()
            # print('Orig loss: {}, Corr loss: {}'.format((u @ (u.T @ grad) - grad).norm().item(), (u @ y - grad).norm().item()))
            # y = self.us[key].T @ grad
            
            # for _ in range(3):
            #     y = y - u.T @ (u @ y - grad)
            k = 30
            comp_grad = u.T @ grad
            decomp_grad = u @ comp_grad
            e = grad - decomp_grad
            # if key == 'layer3.0.conv2.weight':
            #     cnt += 1
            #     if cnt % (30 * 10) == 0 and cnt != 0: 
            #         torch.save(e, 'e.pth')
            #         torch.save(grad, 'grad.pth')
            #         torch.save(decomp_grad, 'decomp_grad.pth')
            #         assert False
            _, indexes = e.abs().topk(k, sorted=False)
            values = e[indexes]
            print(values)
            decomp_e = torch.zeros_like(e).float()
            decomp_e.scatter_(0, indexes, values)
            return comp_grad, decomp_e

    def get_loss_dict(self, glob_dict):
        loc_dict = self.model.state_dict()
        allowed_keys = []
        for k in glob_dict.keys():
            if self.comp_other and not ('bias' in k and ('conv' in k or 'fc' in k)) and 'num_batches_tracked' not in k:
                allowed_keys.append(k)
            if self.comp_bias:
                allowed_keys.append(k)
            if 'weight' in k and ('conv' in k or 'fc' in k or 'downsample.0' in k):
                allowed_keys.append(k)
        allowed_keys = list(set(allowed_keys))
        func = lambda u, grad: torch.sqrt((u @ (u.T @ grad) - grad).norm()) / grad.numel()

        return {k: func(self.us[k], (loc_dict[k] - glob_dict[k]).flatten())
                for k in allowed_keys}
