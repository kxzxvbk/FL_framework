import random
from torch.utils import data
import numpy as np

support_sampling_method = ['iid', 'dirichlet']


class MyDataset(data.Dataset):
    def __init__(self, tot_data, indexes):
        self.tot_data = tot_data
        self.indexes = indexes

    def __getitem__(self, item):
        return self.tot_data[self.indexes[item]]

    def __len__(self):
        return len(self.indexes)


def sample(method, dataset, client_number):
    assert method in support_sampling_method
    if method == 'iid':
        return iid_sampling(dataset, client_number)


def iid_sampling(dataset, client_number):
    num_items = int(len(dataset) / client_number)
    dict_users, all_index = {}, [i for i in range(len(dataset))]
    for i in range(client_number - 1):
        dict_users[i] = random.sample(all_index, num_items)
        all_index = list(set(all_index).difference(set(dict_users[i])))
    dict_users[client_number - 1] = all_index

    return [MyDataset(tot_data=dataset, indexes=dict_users[i]) for i in range(len(dict_users))]


def dirichlet_sampling(dataset, client_number, alpha):
    n_classes = 10
    label_distribution = np.random.dirichlet([alpha] * client_number, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    class_idcs = [np.argwhere(dataset == y).flatten()
                  for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标

    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs
