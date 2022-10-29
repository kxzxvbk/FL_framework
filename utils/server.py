import copy
from utils.client import get_loss


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

    def test(self, model, loss, test_loader=None):
        if test_loader:
            old_loader = self.test_loader
            self.test_loader = test_loader
        model = copy.deepcopy(model)

        correct = 0
        total = 0
        tot_loss = 0
        model.eval()
        model.to(self.device)

        criterion = get_loss(loss)

        for _, (batch_x, batch_y) in enumerate(self.test_loader):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            o = model(batch_x)
            tot_loss += criterion(o, batch_y).item()
            y_pred = o.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(batch_y.data.view_as(y_pred)).long().sum().item()
            total += batch_y.shape[0]
        model.to('cpu')

        avg_acc = correct / total
        avg_loss = tot_loss / total

        if test_loader:
            self.test_loader = old_loader

        res_dict = {'acc': avg_acc, 'loss': avg_loss}

        return res_dict
