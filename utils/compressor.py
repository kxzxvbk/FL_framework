from functools import reduce
import torch
import copy
cnt = 0


# fed avg method
def fed_avg(clients, server, tb_logger):
    total_samples = sum([client.sample_num for client in clients])
    server['glob_dict'] = {k: reduce(lambda x, y: x + y,
                                     [client.sample_num / total_samples * client.model.state_dict()[k]
                                      for client in clients])
                           for k in clients[0].fed_keys}
    trans_cost = 0
    state_dict = clients[0].model.state_dict()
    for k in clients[0].fed_keys:
        trans_cost += len(clients) * state_dict[k].numel()
    # check_avg(clients, server, tb_logger)
    return trans_cost


def check_avg(clients, server, tb_logger):
    global cnt
    cnt += 1
    deal_keys = []
    for k in server['glob_dict'].keys():
        if len(server['glob_dict'][k].shape) in [2, 4]:
            deal_keys.append(k)

    state_dicts = [client.model.state_dict() for client in clients]
    for k in deal_keys:
        check_space_avg(copy.deepcopy([sd[k] for sd in state_dicts]), k, tb_logger)


def check_space_avg(sds, key, tb_logger):
    global_w = reduce(lambda x, y: x + y, [s for s in sds]) / len(sds)
    tot_w = torch.stack([s for s in sds], dim=0)
    # tb_logger.add_histogram('div_' + key, tot_w - global_w, cnt)
    tb_logger.add_scalar('div/' + key, torch.std(tot_w - global_w).item(), cnt)
