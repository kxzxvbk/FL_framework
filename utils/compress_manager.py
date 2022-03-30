from functools import reduce


class CompressManager:
    def __init__(self, clients, server, method='fix_round'):
        self.clients = clients
        self.server = server
        self.method = method

        allowed_list = []
        for k in self.clients[0].fed_keys:
            if self.clients[0].comp_other and not ('bias' in k and ('conv' in k or 'fc' in k)) and 'num_batches_tracked' not in k:
                allowed_list.append(k)
            if self.clients[0].comp_bias:
                allowed_list.append(k)
            if 'weight' in k and ('conv' in k or 'fc' in k or 'downsample.0' in k):
                allowed_list.append(k)
        allowed_list = list(set(allowed_list))

        self.best_loss = {k: -1 for k in allowed_list}
        self.acc_los = {k: 0 for k in allowed_list}
        self.allowed_list = allowed_list
        self.begin_epo = 0

    def judge(self, rnd):
        if self.clients[0].us == {}:
            self.begin_epo = rnd
        if self.method is None or self.method == 'every_round':
            return self.allowed_list
        elif self.method == 'fix_round':
            if (rnd - self.begin_epo) % 10 == 0:
                return self.allowed_list
            return []
        elif self.method == 'ratio':
            for k in self.best_loss:
                if self.best_loss[k] == -1:
                    return list(self.best_loss.keys())
            func = lambda dic1, dic2: {k1: dic1[k1] + dic2[k1] for k1 in dic1}
            ls_dict = reduce(func, [client.get_loss_dict(self.server['glob_dict']) for client in self.clients])

            update_list = []
            for k in ls_dict:
                if ls_dict[k] >= 1.5 * self.best_loss[k]:
                    update_list.append(k)
            return update_list
        elif self.method == 'integration':
            for k in self.best_loss:
                if self.best_loss[k] == -1:
                    return list(self.best_loss.keys())
            func = lambda dic1, dic2: {k1: dic1[k1] + dic2[k1] for k1 in dic1}
            ls_dict = reduce(func, [client.get_loss_dict(self.server['glob_dict']) for client in self.clients])

            for k in self.acc_los:
                self.acc_los[k] += ls_dict[k]

            update_list = []
            for k in ls_dict:
                print(self.acc_los[k], self.best_loss[k], ls_dict[k])
                if self.acc_los[k] >= 5 * self.best_loss[k]:
                    update_list.append(k)

            for k in update_list:
                self.acc_los[k] = 0
            return update_list
        return self.clients[0].fed_keys

    def update_best_loss(self, keys):
        if self.method == 'ratio' or self.method == 'integration':
            func = lambda dic1, dic2: {k: dic1[k] + dic2[k] for k in keys}
            ls_dict = reduce(func, [client.get_loss_dict(self.server['glob_dict']) for client in self.clients])
            self.best_loss.update(ls_dict)
