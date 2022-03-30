import torch
from functools import reduce
from utils.utils import orthogonalize, update_svd
import copy
trans_cost = 0


def comp_cond(k):
    return 'weight' in k and ('conv' in k or 'fc' in k or 'downsample.0' in k)


# fed avg method
def fed_avg(clients, server):
    server['glob_dict'] = {k: reduce(lambda x, y: x + y,
                                     [client.model.state_dict()[k] for client in clients]) / len(clients)
                           for k in clients[0].fed_keys}
    trans_cost = 0
    state_dict = clients[0].model.state_dict()
    for k in clients[0].fed_keys:
        trans_cost += len(clients) * state_dict[k].numel()
    return trans_cost


# top-k
def top_k(clients, server):
    glob_dict = server['glob_dict']
    res_dict = {}
    trans_cost = 0
    
    # only weight tensors are needed to compress
    compress_list = []
    raw_list = []
    for k in clients[0].fed_keys:
        if clients[0].comp_other and not ('bias' in k and ('conv' in k or 'fc' in k)) and 'num_batches_tracked' not in k:
            compress_list.append(k)
        if clients[0].comp_bias:
            compress_list.append(k)
        if comp_cond(k):
            compress_list.append(k)
        elif k not in compress_list:
            raw_list.append(k)
    compress_list = list(set(compress_list))

    for client in clients:
        loc_dict = client.model.state_dict()
        for key in compress_list:
            # calculate grad
            grad = loc_dict[key] - glob_dict[key]

            # compress
            orig_shape = grad.shape
            grad = grad.reshape((grad.shape[0], -1))
            k = max(1, grad.shape[1] // client.compress_ratio)
            values, indexes = grad.topk(k, sorted=False)
            trans_cost += 2 * values.numel()

            # decompress
            decomp_grad = torch.zeros_like(grad).float()
            decomp_grad.scatter_(1, indexes, values)
            decomp_grad = decomp_grad.reshape(orig_shape)

            # aggregate
            if key in res_dict:
                res_dict[key] += decomp_grad
            else:
                res_dict[key] = decomp_grad
    for k in res_dict:
        res_dict[k] /= len(clients)

    # aggregating none weight tensors
    server['glob_dict'].update({k: reduce(lambda x, y: x + y,
                                          [client.model.state_dict()[k] for client in clients]) / len(clients)
                                for k in raw_list})

    server.apply_grad(res_dict)
    return trans_cost


# qsgd
def qsgd(clients, server, quantum_num=127, eps=1e-6):
    glob_dict = server['glob_dict']
    res_dict = {}

    # only weight tensors are needed to compress
    compress_list = []
    raw_list = []
    for k in clients[0].fed_keys:
        if clients[0].comp_other and not ('bias' in k and ('conv' in k or 'fc' in k)) and 'num_batches_tracked' not in k:
            compress_list.append(k)
        if clients[0].comp_bias:
            compress_list.append(k)
        if comp_cond(k):
            compress_list.append(k)
        elif k not in compress_list:
            raw_list.append(k)
    compress_list = list(set(compress_list))
    trans_cost = 0

    for client in clients:
        loc_dict = client.model.state_dict()
        for k in compress_list:
            # calculate grad
            grad = loc_dict[k] - glob_dict[k]

            # compress
            orig_shape = grad.shape
            grad = grad.flatten()
            norm = grad.norm().item()
            abs_grad = grad.abs()

            level_float = quantum_num / (norm * abs_grad + eps)
            pre_level = level_float.floor()
            prob = torch.empty_like(grad).uniform_()
            is_next_level = (prob < (level_float - pre_level)).type(torch.float32)
            new_level = (pre_level + is_next_level)

            sign = grad.sign()
            comp_grad = (new_level * sign).type(torch.int16)
            comp_grad = comp_grad.type(torch.int8 if quantum_num < 128 else torch.half)

            # decompress
            decomp_grad = comp_grad.type(torch.float32)
            decomp_grad = (norm / quantum_num * decomp_grad).reshape(orig_shape)

            # aggregate
            if k in res_dict:
                res_dict[k] += decomp_grad
            else:
                res_dict[k] = decomp_grad

    for k in res_dict:
        res_dict[k] /= len(clients)
    
    # aggregating none weight tensors
    server['glob_dict'].update({k: reduce(lambda x, y: x + y,
                                          [client.model.state_dict()[k] for client in clients]) / len(clients)
                                for k in raw_list})

    server.apply_grad(res_dict)
    return trans_cost


def atomo(clients, server):
    glob_dict = server['glob_dict']
    res_dict = {}
    trans_cost = 0
    
    # only weight tensors are needed to compress
    compress_list = []
    raw_list = []
    for k in clients[0].fed_keys:
        if clients[0].comp_other and not ('bias' in k and ('conv' in k or 'fc' in k)) and 'num_batches_tracked' not in k:
            compress_list.append(k)
        if clients[0].comp_bias:
            compress_list.append(k)
        if comp_cond(k):
            compress_list.append(k)
        elif k not in compress_list:
            raw_list.append(k)
    compress_list = list(set(compress_list))

    for client in clients:
        loc_dict = client.model.state_dict()
        for key in compress_list:
            # calculate grad
            grad = loc_dict[key] - glob_dict[key]

            # compress
            orig_shape = grad.shape
            k = max(1, grad.shape[1] // client.compress_ratio)
            u, s, v = torch.svd(grad)
            u = u[:, :k]
            comp_grad = u.T @ grad

            # decompress
            decomp_grad = u @ comp_grad

            # aggregate
            if key in res_dict:
                res_dict[key] += decomp_grad
            else:
                res_dict[key] = decomp_grad
    for k in res_dict:
        res_dict[k] /= len(clients)

    # aggregating none weight tensors
    server['glob_dict'].update({k: reduce(lambda x, y: x + y,
                                          [client.model.state_dict()[k] for client in clients]) / len(clients)
                                for k in raw_list})

    server.apply_grad(res_dict)
    return trans_cost



# powersgd
def powersgd(clients, server):
    glob_dict = server['glob_dict']
    p_dict = {}
    m_dict = {}
    shape_dict = {}
    trans_cost = 0
    # calculate P
    for k in clients[0].fed_keys:
        if not ((('conv' in k or 'fc' in k) and 'weight' in k) or 'downsample.0' in k):
            continue
        ps = []
        ms = []
        for client in clients:
            grad = client.model.state_dict()[k] - glob_dict[k]
            shape_dict[k] = grad.shape
            grad = grad.reshape(grad.shape[0], -1)
            q = client.qs[k]

            p = grad @ q
            ps.append(p)
            ms.append(grad)
        p_dict[k] = ps
        m_dict[k] = ms

    # all reduce & orth for P
    for k in p_dict.keys():
        p_dict[k] = orthogonalize(reduce(lambda x, y: x + y, p_dict[k]) / len(clients))
        m_dict[k] = reduce(lambda x, y: x + y, m_dict[k]) / len(clients)
        trans_cost += m_dict[k][0].numel() * len(clients)

    # update Q
    q_dict = {}
    for k in p_dict.keys():
        q_dict[k] = m_dict[k].T @ p_dict[k]
        trans_cost += len(clients) * q_dict[k].numel()

    # mean all reduce for Q
    for client in clients:
        client.qs = copy.deepcopy(q_dict)

    # decompress
    res_dict = {}
    for k in clients[0].fed_keys:
        if k in p_dict.keys():
            res_dict[k] = (p_dict[k] @ q_dict[k].T).reshape(shape_dict[k])
        else:
            res_dict[k] = reduce(lambda x, y: x + y, [client.model.state_dict()[k] for client in clients]) / len(
                clients)
            trans_cost += len(clients) * clients[0].model.state_dict()[k].numel()

    server.apply_grad(res_dict)
    return trans_cost


def broad_cast_u(clients, u_dict):
    for client in clients:
        client.set_u_dict(u_dict)


# fed comp stage 1
def stage1(clients, server, keys, **kwargs):
    if not clients[0].increment_svd or kwargs['u_dict'] is None:
        return _no_increment_svd(clients, server, keys)
    return _increment_svd(clients, server, keys, u_dict=kwargs['u_dict'], s_dict=kwargs['s_dict'], v_dict=kwargs['v_dict'])


def cut_u(u_dict, s_dict):
    cuted_u_dict = {}
    for k in u_dict.keys():
        u = u_dict[k]
        s = s_dict[k]
        thresh = s.max().item() / 3
        indexes = []
        for i in range(s.shape[0]):
            if s[i] < thresh:
                indexes.append(i)
        cuted_u_dict[k] = u[:, indexes]
    return cuted_u_dict


def _no_increment_svd(clients, server, keys):
    glob_dict = server['glob_dict']
    u_dict = {}
    s_dict = {}
    v_dict = {}
    trans_cost = 0

    for k in keys:
        # calculate grad
        weights = []
        for client in clients:
            grad = client.model.state_dict()[k] - glob_dict[k]
            trans_cost += grad.numel()
            weights.append(grad.flatten().reshape((-1, 1)))
        weights = torch.cat(weights, dim=1)
        u, s, v = torch.svd(weights)

        u_dict[k] = u
        s_dict[k] = s
        v_dict[k] = v

    return trans_cost, u_dict, s_dict, v_dict


def _increment_svd(clients, server, keys, **kwargs):
    glob_dict = server['glob_dict']
    u_dict = {}
    s_dict = {}
    v_dict = {}
    trans_cost = 0
    old_u_dict = kwargs['u_dict']
    old_s_dict = kwargs['s_dict']
    old_v_dict = kwargs['v_dict']

    for k in keys:
        # calculate grad
        weights = []
        u, s, v = old_u_dict[k], old_s_dict[k], old_v_dict[k]
        assert u.shape[1] == 30
        for client in clients:
            u, s, v = update_svd(u, s, v, client.model.state_dict()[k].flatten(), True)

        u_dict[k] = u
        s_dict[k] = s
        v_dict[k] = v

    return trans_cost, u_dict, s_dict, v_dict


def stage2(clients, server, error_corr):
    global trans_cost
    glob_dict = server['glob_dict']
    u_dict = clients[0].us
    trans_cost = 0

    # only weight tensors are needed to compress
    compress_list = []
    raw_list = []
    for k in clients[0].fed_keys:
        if clients[0].comp_other and not ('bias' in k and ('conv' in k or 'fc' in k)) and 'num_batches_tracked' not in k:
            compress_list.append(k)
        if clients[0].comp_bias:
            compress_list.append(k)
        if comp_cond(k):
            compress_list.append(k)
        elif k not in compress_list:
            raw_list.append(k)
    compress_list = list(set(compress_list))

    # aggregating weight tensors
    def _decompress(gradient_error, key):
        global trans_cost
        gradient, error = gradient_error
        rank = gradient.shape[0]
        trans_cost += gradient.numel()
        return u_dict[key][:, :rank] @ gradient + error

    res_dict = {k: reduce(lambda x, y: x + y,
                          [_decompress(client.compress(glob_dict[k], k, error_corr), k).reshape(glob_dict[k].shape)
                           for client in clients]) / len(clients)
                for k in compress_list
                }

    # aggregating none weight tensors
    server['glob_dict'].update({k: reduce(lambda x, y: x + y,
                                          [client.model.state_dict()[k] for client in clients]) / len(clients)
                                for k in raw_list})

    # DEBUGGING
    # res_dict1 = {k: reduce(lambda x, y: x + y,
    #                      [client.model.state_dict()[k] for client in clients]) / len(clients) - glob_dict[k]
    #              for k in compress_list}
    
    # with open('loss_file_ec', 'a+') as f:
    #     for k in res_dict1:
    #         ls = torch.sqrt((res_dict1[k] - res_dict[k]).norm()) / res_dict1[k].numel()
    #         print(k + str(ls))
    #         f.write(k + str(ls) + '\n')
    #     f.write('################################################################################\n')
    # DEBUGGING END
    # apply gradient
    server.apply_grad(res_dict)

    for k in raw_list:
        print(k + ': ' + str(glob_dict[k].numel()))
        trans_cost += glob_dict[k].numel() * len(clients)

    return trans_cost
