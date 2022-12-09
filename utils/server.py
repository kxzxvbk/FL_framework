import copy
from functools import reduce

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

    def check_cent(self,  model):
        # used for checking difference for features calculated on different clients.
        # return a diction consisting: abs_value, cos_distance, l2_distance.
        # All returned keys will be later added to tb. Only support model having: compute_feature.
        model.init_eval()
        model = copy.deepcopy(model).cuda()

        augmentation = [transforms.ToTensor(), transforms.Normalize(
            mean=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
            std=[0.24703225141799082, 0.24348516474564, 0.26158783926049628])]
        test_dataset = datasets.CIFAR10(os.path.join('./data', 'CIFAR10'), train=False,
                                        download=True, transform=transforms.Compose(augmentation))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=256, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=False)

        tot_feature = []
        tot_label = []

        for _, (batch_x, batch_y) in enumerate(test_loader):
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            tot_feature.append(torch.nn.functional.normalize(model.forward_eval(batch_x)[0].detach(), dim=1))
            tot_label.append(batch_y)

        tot_feature = torch.cat(tot_feature, dim=0)
        tot_label = torch.cat(tot_label, dim=0)

        class_indexes = [torch.where(tot_label == y, 1, 0).flatten() for y in range(10)]
        class_feature = [tot_feature[class_indexes[y]] for y in range(10)]

        i_dists = []
        o_dists = []
        for i in range(10):
            # mean std min max
            this = class_feature.pop(i)  # B * C
            other = torch.cat(class_feature, dim=0)  # B * C

            inner = this @ this.T
            outer = this @ other.T

            i_dist = {'inner_mean': torch.mean(inner).item(), 'inner_std': torch.std(inner).item(),
                      'inner_min': torch.min(inner).item(), 'inner_max': torch.max(inner).item()}
            o_dist = {'outer_mean': torch.mean(outer).item(), 'outer_std': torch.std(outer).item(),
                      'outer_min': torch.min(outer).item(), 'outer_max': torch.max(outer).item()}
            i_dists.append(i_dist)
            o_dists.append(o_dist)
            class_feature.insert(i, this)

        res_dict = {'hist': torch.histc(tot_feature) / tot_feature.numel()}
        for k in i_dists[0]:
            res_dict[k] = sum([t[k] for t in i_dists]) / len(i_dists)

        for k in o_dists[0]:
            res_dict[k] = sum([t[k] for t in o_dists]) / len(o_dists)

        model.to('cpu')

        return res_dict

    def check_bias(self, client_pool, save_ram=False, max_client=float('inf')):
        # used for checking difference for features calculated on different clients.
        # return a diction consisting: abs_value, cos_distance, l2_distance.
        # All returned keys will be later added to tb. Only support model having: compute_feature.
        model_list = [copy.deepcopy(client.model) for client in client_pool]
        max_len = min(max_client, len(model_list))
        model_list = model_list[:max_len]

        # to device
        for model in model_list:
            model.eval()
            if not save_ram:
                model.to(self.device)

        abs_vals = []
        l2_dists = []
        cos_dists = []

        augmentation = [transforms.ToTensor(), transforms.Normalize(
            mean=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
            std=[0.24703225141799082, 0.24348516474564, 0.26158783926049628])]
        test_dataset = datasets.CIFAR10(os.path.join('./data', 'CIFAR10'), train=False,
                                        download=True, transform=transforms.Compose(augmentation))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=256, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=False)

        for _, (batch_x, batch_y) in enumerate(test_loader):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            batch_feature = []
            for model in model_list:
                if save_ram:
                    model = model.to(self.device)
                    batch_feature.append(model.compute_feature(batch_x).detach())
                    model = model.to('cpu')
                else:
                    batch_feature.append(model.compute_feature(batch_x).detach())
            abs_val = torch.cat([torch.abs(batch_feature[ii]) for ii in range(len(batch_feature))], dim=0)
            l2_dist = torch.cat([torch.norm(batch_feature[i] - batch_feature[i + 1], dim=-1, keepdim=False)
                                 for i in range(len(batch_feature) - 1)], dim=0)
            batch_feature = torch.nn.functional.normalize(torch.stack(batch_feature, dim=0), dim=-1)
            cos_dist = torch.cat([torch.diag(batch_feature[i] @ batch_feature[i+1].T)
                                  for i in range(len(batch_feature) - 1)])

            abs_vals.append(abs_val.cpu())
            l2_dists.append(l2_dist.cpu())
            cos_dists.append(cos_dist.cpu())
        abs_vals = torch.cat(abs_vals, dim=0)
        l2_dists = torch.cat(l2_dists, dim=0)
        cos_dists = torch.cat(cos_dists, dim=0)

        res_dict = {}
        res_dict['abs_mean'] = torch.mean(abs_vals).item()
        res_dict['abs_std'] = torch.std(abs_vals).item()
        res_dict['abs_max'] = torch.max(abs_vals).item()

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

    def test(self, model, test_loader=None, train_epoch=3):
        if test_loader:
            old_loader = self.test_loader
            self.test_loader = test_loader
        model = copy.deepcopy(model)

        criterion = nn.CrossEntropyLoss()
        model.init_eval()
        model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), 0.01,
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
                o, _ = model.forward_eval(batch_x)
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

            tot_info = {}
            cnt = 0

            for _, (batch_x, batch_y) in enumerate(test_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                o, info = model.forward_eval(batch_x)

                cnt += 1
                for k in info:
                    if k not in tot_info:
                        tot_info[k] = info[k]
                    else:
                        if 'max' in k:
                            tot_info[k] = max(info[k], tot_info[k])
                        else:
                            tot_info[k] += info[k]

                tot_loss += criterion(o, batch_y).item()
                y_pred = o.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(batch_y.data.view_as(y_pred)).long().sum().item()
                total += batch_y.shape[0]

            avg_acc = correct / total
            avg_loss = tot_loss / total

            print('linear protocol: ' + str(avg_acc) + '      ' + str(avg_loss))
        for k in tot_info:
            if 'max' not in k:
                tot_info[k] /= cnt
        tot_info['acc'] = avg_acc
        tot_info['loss'] = avg_loss
        return tot_info
