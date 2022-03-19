import torch
import copy
import pickle


def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    # using Schmidt orth method on matrix
    n, m = matrix.shape
    matrix = copy.deepcopy(matrix)
    for i in range(m):
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            rest -= torch.sum(col * rest, dim=0) * col
    return matrix


def get_params_number(net):
    # get total params number in nn.Module net
    res = 0
    for param in net.parameters():
        res += param.numel()
    return res


def update_svd(u, s, v, a, keep_rank=True):
    b = torch.zeros(s.shape[0] + 1)
    b[-1] = 1

    m = u.T @ a
    p = a - (u @ m)
    ra = torch.sqrt(p.T @ p)
    p = p / ra

    v = torch.vstack([v, torch.zeros(v.shape[0])])
    n = v.T @ b
    q = b - v @ n
    rb = torch.sqrt(q.T @ q)
    q = q / rb

    k = torch.diag(torch.zeros(s.shape[0] + 1))
    k[:-1, -1] = m
    k[:-1, :-1] = torch.diag(s)
    k[-1, -1] = ra

    u_p = torch.vstack([u.T, p]).T
    v_q = torch.vstack([v.T, q]).T

    u1, s1, v1 = torch.svd(k)

    u_new = u_p @ u1
    v_new = v_q @ v1

    if keep_rank:
        s1 = s1[:-1]
        u_new = u_new[:, :-1]
        v_new = v_new[:-1, :-1]

    return u_new, s1, v_new


def save_file(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def cont2dis(arr, mini=None, maxi=None, bins=32):
    # 将向量中的连续值离散化，变成bins个离散的值
    if not maxi:
        maxi = torch.max(arr).item()
    if not mini:
        mini = torch.min(arr).item()
    lent = (maxi - mini) / bins
    ret_arr = []
    for j in range(len(arr)):
        ret_arr.append((arr[j].item() - mini) // lent)
    ret_arr = torch.tensor(ret_arr).int()
    return ret_arr


def entropy(labels, base=None):
    # 计算labels的熵
    maxi = torch.max(labels).item()
    probs = torch.zeros(maxi + 1)
    for j in range(len(labels)):
        probs[labels[j]] += 1
    probs = probs / probs.sum()
    base = torch.tensor(2)
    res = 0
    for j in range(len(probs)):
        if probs[j] != 0:
            res -= probs[j] * torch.log(probs[j]) / torch.log(base)
    return res


def u_entropy(labels1, labels2):
    # 联合熵E（labels1，labels2）
    maxi = max(torch.max(labels1).item(), torch.max(labels2).item())
    probs = torch.zeros((maxi + 1, maxi + 1))
    for j in range(len(labels1)):
        probs[labels1[j]][labels2[j]] += 1
    base = torch.tensor(2)
    probs = probs / probs.sum()
    res = 0
    for j in range(probs.shape[0]):
        for k in range(probs.shape[1]):
            if probs[j][k] != 0:
                res -= probs[j][k] * torch.log(probs[j][k]) / torch.log(base)
    return res


def t_entropy(labels1, labels2):
    # 条件熵E(labels1|labels2)
    maxi = max(torch.max(labels1).item(), torch.max(labels2).item())
    counts = torch.zeros((maxi + 1, maxi + 1))
    for j in range(len(labels1)):
        counts[labels1[j]][labels2[j]] += 1
    probs1 = counts / counts.sum()
    probs2 = torch.zeros_like(counts)
    for j in range(probs2.shape[1]):
        if counts[:, j].sum() != 0:
            probs2[:, j] = counts[:, j] / counts[:, j].sum()

    base = torch.tensor(2)
    res = 0.
    for j in range(probs1.shape[0]):
        for k in range(probs1.shape[1]):
            if probs2[j][k] != 0:
                res -= probs1[j][k] * torch.log(probs2[j][k]) / torch.log(base)
    return res


def m_entropy(labels1, labels2):
    # 互信息I（labels1， labels2）
    return entropy(labels1) + entropy(labels2) - u_entropy(labels1, labels2)


def batch_entropy(batch):
    batch = collate(batch)
    ents = {}
    for key in batch.keys():
        sub_batch = torch.stack(batch[key], dim=0)
        maxi = torch.max(sub_batch).item()
        mini = torch.min(sub_batch).item()
        ents[key] = {i: entropy(cont2dis(sub_batch[i], mini=mini, maxi=maxi)).item() for i in range(sub_batch.shape[0])}
    return ents


def union_entropy(batch):
    ents = {}
    for i in range(len(batch)):
        batch_i = batch[i]
        concat_batch = []
        for key in batch_i.keys():
            concat_batch.append(batch_i[key])
        concat_batch = torch.stack(concat_batch, dim=0)
        ents[i] = m_entropy(cont2dis(concat_batch[0], mini=torch.min(concat_batch), maxi=torch.max(concat_batch)),
                            cont2dis(concat_batch[1], mini=torch.min(concat_batch), maxi=torch.max(concat_batch))).item()
    return ents


def spatial_entropy(batch):
    batch = collate(batch)
    ents = {}
    for key in batch.keys():
        sub_batch = torch.stack(batch[key], dim=0)
        maxi = torch.max(sub_batch).item()
        mini = torch.min(sub_batch).item()
        ents[key] = {i: m_entropy(cont2dis(sub_batch[i], mini=mini, maxi=maxi),
                                  cont2dis(sub_batch[i + 1], mini=mini, maxi=maxi)).item()
                     for i in range(sub_batch.shape[0] - 1)}
    return ents


def temporal_entropy(batch1, batch2):
    batch1 = collate(batch1)
    batch2 = collate(batch2)
    ents = {}
    for key in batch1.keys():
        sub_batch1 = torch.stack(batch1[key], dim=0)
        sub_batch2 = torch.stack(batch2[key], dim=0)
        maxi = max(torch.max(sub_batch1).item(), torch.max(sub_batch2).item())
        mini = min(torch.min(sub_batch1).item(), torch.min(sub_batch2).item())
        ents[key] = {i: m_entropy(cont2dis(sub_batch1[i], mini=mini, maxi=maxi),
                                  cont2dis(sub_batch2[i], mini=mini, maxi=maxi)).item()
                     for i in range(sub_batch1.shape[0])}
    return ents


def collate(batch):
    ret = {}
    for key in batch[0].keys():
        ret[key] = [b[key] for b in batch]
    return ret
