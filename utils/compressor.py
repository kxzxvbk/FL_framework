from functools import reduce
trans_cost = 0


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
