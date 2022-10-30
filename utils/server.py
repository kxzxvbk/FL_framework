import copy
from utils.client import get_loss
import torch


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

    def check_bias(self, client_pool, save_ram=False, max_client=float('inf')):
        # only support model having: compute_feature
        model_list = [copy.deepcopy(client.model) for client in client_pool]
        max_len = min(max_client, len(model_list))
        model_list = model_list[:max_len]

        # to device
        for model in model_list:
            model.eval()
            if not save_ram:
                model.to(self.device)

        l2_dists = []
        cos_dists = []

        for _, (batch_x, batch_y) in enumerate(self.test_loader):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            batch_feature = []
            for model in model_list:
                if save_ram:
                    model = model.to(self.device)
                    batch_feature.append(model.compute_feature(batch_x).detach())
                    model = model.to('cpu')
                else:
                    batch_feature.append(model.compute_feature(batch_x).detach())

            l2_dist = torch.cat([torch.norm(batch_feature[i] - batch_feature[i + 1], dim=-1, keepdim=False)
                                 for i in range(len(batch_feature) - 1)], dim=0)
            batch_feature = torch.nn.functional.normalize(torch.stack(batch_feature, dim=0), dim=-1)
            cos_dist = torch.cat([torch.diag(batch_feature[i] @ batch_feature[i+1].T)
                                  for i in range(len(batch_feature) - 1)])
            l2_dists.append(l2_dist.cpu())
            cos_dists.append(cos_dist.cpu())

        l2_dists = torch.cat(l2_dists, dim=0)
        cos_dists = torch.cat(cos_dists, dim=0)
        res_dict = {}
        res_dict['l2_mean'] = torch.mean(l2_dists).item()
        res_dict['l2_std'] = torch.std(l2_dists).item()
        res_dict['l2_max'] = torch.max(l2_dists).item()

        res_dict['cos_mean'] = torch.mean(cos_dists).item()
        res_dict['cos_std'] = torch.std(cos_dists).item()
        res_dict['cos_min'] = torch.min(cos_dists).item()

        if not save_ram:
            for model in model_list:
                model.to('cpu')
        return res_dict

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
