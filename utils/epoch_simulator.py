from utils.client import Client, get_optimizer
from utils.dataset import DatasetConstructor
from utils.sampling import sample
from utils.client_pool import ClientPool
from utils.logger import Logger
from utils.server import Server
from utils.utils import get_params_number
import numpy as np
from torch.utils import data
import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import os


class Simulator:
    """
    a simulator for a single device to perform federated learning
    """

    def __init__(self, args):
        self.args = args

    def run(self):
        client_pool = ClientPool(self.args)
        tb_logger = SummaryWriter(self.args.logging_path)
        logger = Logger(self.args.logging_path)

        # load dataset
        train_set = DatasetConstructor(self.args).get_dataset()
        test_set = DatasetConstructor(self.args).get_dataset(train=False)

        # split dataset into clients. alpha affects the distribution for dirichlet non-iid sampling.
        # If you don't use dirichlet, this parameter can be omitted.
        train_sets = sample(self.args.sample_method, train_set, self.args.client_num, alpha=self.args.alpha)

        # if you need all clients to test locally use next line to split test sets
        # test_sets = sample(self.args.sample_method, test_set, self.args.client_num)

        # initialize clients, assemble datasets
        for i in range(self.args.client_num):
            client_pool.append(Client(train_sets[i], args=self.args, client_id=i, test_dataset=None))
        print(client_pool[0].model)
        logger.logging('All clients initialized.')
        logger.logging('Parameter number in each model: {:.2f}M'
                       .format(get_params_number(client_pool[0].model) / 1e6))
        # global initialization
        if self.args.fed_dict == 'all':
            glob_dict = client_pool[0].model.state_dict()
        elif self.args.fed_dict == 'except_bn':
            state_dict = client_pool[0].model.state_dict()
            glob_dict = {}
            for key in state_dict:
                if 'downsample.1' not in key and 'bn' not in key:
                    glob_dict[key] = state_dict[key]
        else:
            glob_dict = client_pool[0].get_state_dict(self.args.fed_dict)
        if self.args.resume:
            glob_dict = torch.load('./model_checkpoints/iid.ckpt')

        server = Server(self.args.device, data.DataLoader(test_set, batch_size=32, shuffle=True))
        server.add_attr(name='glob_dict', item=glob_dict)
        client_pool.server = server

        # set fed keys in each client and init compression settings
        client_pool.set_fed_keys()
        client_pool.sync()
        # for evaluation
        # test_acc, test_loss = server.test(model=client_pool[0].model, train_epoch=30)
        # assert False

        train_accuracies = []
        train_losses = []
        test_accuracies = []
        test_losses = []
        trans_costs = []

        # training loop
        for i in range(self.args.start_round, self.args.glob_eps):
            train_acc = {1: 0, 5: 0}
            train_loss = 0
            total_client = 0
            print('Starting round: ' + str(i))

            participated_clients = np.array(range(self.args.client_num))
            participated_clients = sorted(list(np.random.choice(participated_clients,
                                                                int(self.args.client_sample_rate *
                                                                    participated_clients.shape[0]), replace=False)))
            optimizers = [get_optimizer(name=self.args.optimizer, lr=self.args.lr * (self.args.decay_factor ** i),
                                        momentum=self.args.momentum, weights=client_pool[j].model.parameters())
                          for j in participated_clients]

            for j in participated_clients:
                client_pool[j].model.reset_encoder_k()

            for ep in tqdm.tqdm(range(self.args.loc_eps)):
                for j in participated_clients:
                    total_client += 1
                    client = client_pool[j]
                    acc, loss = client.train_epoch(optimizers[j], self.args.loss)
                    for k in acc:
                        train_acc[k] += acc[k]
                    train_loss += loss
            for k in train_acc:
                train_acc[k] /= self.args.loc_eps
            train_loss /= self.args.loc_eps

            # test for client 0
            if i % self.args.test_freq == 0:
                info = server.test(model=client_pool[0].model, train_epoch=30)
                for k in info:
                    tb_logger.add_scalar('test_client0/{}'.format(k), info[k], i)

            # aggregation and sync
            trans_cost = client_pool.aggregate(i, )

            logger.logging('epoch:{}, train_acc@1: {:.4f}, train_acc@5: {:.4f}, train_loss: {:.4f}, trans_cost: {:.4f}M'
                           .format(i, train_acc[1] / len(participated_clients),
                                   train_acc[5] / len(participated_clients),
                                   train_loss / len(participated_clients), trans_cost / 1e6))
            tb_logger.add_scalar('train/top1_acc', train_acc[1] / len(participated_clients), i)
            tb_logger.add_scalar('train/top5_acc', train_acc[5] / len(participated_clients), i)
            tb_logger.add_scalar('train/loss', train_loss / len(participated_clients), i)

            if i % self.args.test_freq == 0:
                info = server.test(model=client_pool[0].model, train_epoch=30)
                test_acc, test_loss = info['acc'], info['loss']
                if not os.path.exists('./model_checkpoints'):
                    os.makedirs('./model_checkpoints')
                if len(test_accuracies) == 0 or max(test_accuracies) <= test_acc:
                    torch.save(client_pool.server['glob_dict'], self.args.model_path)
                test_losses.append(test_loss)
                test_accuracies.append(test_acc)
                logger.logging('epoch:{}, test_acc: {:.4f}, test_loss: {:.4f}'
                               .format(i, test_accuracies[-1], test_losses[-1]))
                for k in info:
                    tb_logger.add_scalar('test/{}'.format(k), info[k], i)

                info = server.test(model=client_pool[0].model, train_epoch=50)
                test_acc, test_loss = info['acc'], info['loss']
                if not os.path.exists('./model_checkpoints'):
                    os.makedirs('./model_checkpoints')
                if len(test_accuracies) == 0 or max(test_accuracies) <= test_acc:
                    torch.save(client_pool.server['glob_dict'], self.args.model_path)

                logger.logging('epoch:{}, test_acc: {:.4f}, test_loss: {:.4f}'
                               .format(i, test_accuracies[-1], test_losses[-1]))
                for k in info:
                    tb_logger.add_scalar('test/{}'.format(k), info[k], i)
                # if you want to test all the training set, use following code.
                # BE AWARE: the
                # test_acc, test_loss = server.test(model=client_pool[0].model,
                #                                   loss=self.args.loss,
                #                                   test_loader=data.DataLoader(train_set, batch_size=32, shuffle=True))
                # logger.logging('epoch:{}, test_acc: {:.4f}, test_loss: {:.4f}'
                #                .format(i, test_acc, test_loss))

        info = server.test(model=client_pool[0].model, train_epoch=30)
        test_acc, test_loss = info['acc'], info['loss']
        logger.logging('Final evaluation, test_acc: {:.4f}, test_loss: {:.4f}'
                       .format(test_acc, test_loss))
        np.savez('results', np.array(train_accuracies),
                 np.array(train_losses),
                 np.array(test_accuracies),
                 np.array(test_losses))
