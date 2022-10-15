from models.models import ModelConstructor
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import copy
import torch

cnt = 0


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_optimizer(name, lr, momentum, weights):
    if name == 'sgd':
        return optim.SGD(params=weights, momentum=momentum, lr=lr, weight_decay=1e-4)
    elif name == 'adam':
        return optim.AdamW(params=weights, lr=lr, weight_decay=1e-4)
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
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                           shuffle=True, drop_last=True)
        self.model = ModelConstructor(args).get_model(client_id)
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
        self.model.reset_encoder_k()
        self.model.train()
        self.model.to(self.device)

        accs = {1: [], 5: []}
        tot_loss = []

        op = get_optimizer(name=optimizer, lr=lr, momentum=momentum, weights=self.model.parameters())
        criterion = get_loss(loss)

        for epoch in range(local_eps):
            for i, (images, _) in enumerate(self.train_dataloader):
                images[0] = images[0].to(self.device)
                images[1] = images[1].to(self.device)

                # compute output
                output, target = self.model(im_q=images[0], im_k=images[1])
                loss = criterion(output, target)

                # acc1/acc5 are (K+1)-way contrast classifier accuracy
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                accs[1].append(acc1.item())
                accs[5].append(acc5.item())
                tot_loss.append(loss.item())

                # compute gradient and do SGD step
                op.zero_grad()
                loss.backward()
                op.step()

        avg_acc = {1: sum(accs[1]) / len(accs[1]), 5: sum(accs[5]) / len(accs[5])}
        avg_loss = sum(tot_loss) / len(tot_loss)
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
