from utils.compressor import *
from utils.compress_manager import CompressManager
from utils.utils import save_file, load_file, batch_entropy, spatial_entropy, temporal_entropy, union_entropy
import time
import os


class ClientPool:
    """
    A container to hold clients.
    """

    def __init__(self, args):
        self.clients = []
        self.server = None
        self.method = None
        self.args = args

        # only used in fedcomp
        self.compress_manager = None
        self.u_dict = None
        self.s_dict = None
        self.v_dict = None

        self.last_grads = None

        if args.resume:
            self.u_dict = load_file('./model_checkpoints/u_dict.pkl')
            self.s_dict = load_file('./model_checkpoints/s_dict.pkl')
            self.v_dict = load_file('./model_checkpoints/v_dict.pkl')

    def append(self, item):
        """
        append: add a client into the container
        :param item: the client to be added
        :return: None
        """
        self.clients.append(item)

    def aggregate(self, train_round, manager_method=None):
        """
        aggregate: applying a aggregation method to update the global model
        :return: None
        """
        if self.args.warming_up and train_round < 3 and self.method != 'fedcomp':
            trans_cost = fed_avg(self.clients, self.server)
            self.sync()
        elif self.args.warming_up and train_round < 2 and self.method == 'fedcomp':
            trans_cost = fed_avg(self.clients, self.server)
            self.sync()
        elif self.method == 'avg':
            glob_dict = self.server['glob_dict']
            grads = []
            # for i in range(2):
            #     client = self.clients[i]
            #     states = client.model.state_dict()
            #     lis = []
            #     for k in client.fed_keys:
            #         if 'weight' in k and ('conv' in k or 'fc' in k or 'downsample.0' in k):
            #             lis.append(k)
            #     grad_dict = {k: (states[k] - glob_dict[k]).flatten() for k in lis}
                # grad_dict = {k: torch.randn(states[k].shape).flatten() for k in ['conv1.weight']}
            #     grads.append(grad_dict)
            # ents = {}
            # print('start')
            # ents['entros'] = batch_entropy(grads)
            # print('finish')
            # ents['spatial'] = spatial_entropy(grads)
            # print('finish')
            # ents['temporal'] = temporal_entropy(grads, self.last_grads) if self.last_grads else None
            # print('finish')
            # ents['union entropy'] = union_entropy(grads)
            # ents['iteration'] = train_round
            # self.last_grads = grads
            # with open('entropy_file.txt', 'a+') as f:
            #     f.write(str(ents) + '\n')

            trans_cost = fed_avg(self.clients, self.server)
            self.sync()
        elif self.method == 'topk':
            trans_cost = top_k(self.clients, self.server)
            self.sync()
        elif self.method == 'qsgd':
            trans_cost = qsgd(self.clients, self.server)
            self.sync()
        elif self.method == 'powersgd':
            trans_cost = powersgd(self.clients, self.server)
            self.sync()
        elif self.method == 'fedcomp':
            if self.compress_manager is None:
                self.compress_manager = CompressManager(self.clients, self.server, manager_method)
            lis = self.compress_manager.judge(train_round)
            print('Update keys: ' + str(lis))
            _, u_dict, s_dict, v_dict = stage1(self.clients, self.server, lis, u_dict=self.u_dict, s_dict=self.s_dict, v_dict=self.v_dict)

            # if the system is first called, use fedavg to initialize
            if self.u_dict is None:
                self.u_dict = u_dict
                self.s_dict = s_dict
                self.v_dict = v_dict
                broad_cast_u(self.clients, u_dict=cut_u(u_dict,  s_dict))
                trans_cost = fed_avg(self.clients, self.server)
                self.sync()
                return trans_cost
            if self.clients[0].us == {}:
                broad_cast_u(self.clients, u_dict=cut_u(u_dict, s_dict))
            trans_cost = stage2(self.clients, self.server, self.args.error_corr)
            self.compress_manager.update_best_loss(lis)
            # sync for u_dict
            self.u_dict = u_dict
            broad_cast_u(self.clients, u_dict=u_dict)
            self.sync()
            save_file(self.u_dict, './model_checkpoints/u_dict.pkl')
            save_file(self.v_dict, './model_checkpoints/v_dict.pkl')
            save_file(self.s_dict, './model_checkpoints/s_dict.pkl')
            return trans_cost
        else:
            print('Unrecognized compression method: ' + self.method)
            assert False
        return trans_cost

    def flush(self):
        """
        flush all the clients
        :return: None
        """
        self.clients = []

    def __getitem__(self, item):
        return self.clients[item]

    def sync(self):
        """
        given a global state_dict, require all clients' model is set equal as server
        :return: None
        """
        state_dict = self.server['glob_dict']
        for client in self.clients:
            client.update_model(state_dict)

    def set_up_fed_keys(self, keys):
        for client in self.clients:
            client.set_fed_keys(keys)

    def setup_compression_settings(self, method='avg', compress_ratio=40):
        self.method = method
        if method == 'powersgd':
            for client in self.clients:
                client.setup_powersgd(compress_ratio)

    def set_fed_keys(self):
        for client in self.clients:
            client.set_fed_keys(self.server['glob_dict'].keys())
