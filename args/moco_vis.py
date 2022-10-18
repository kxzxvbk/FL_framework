import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--loc_eps', type=int, default=10, help="rounds of training")
    parser.add_argument('--glob_eps', type=int, default=200, help="global training round")
    parser.add_argument('--client_num', type=int, default=10, help="number of client")
    parser.add_argument('--client_sample_rate', type=float, default=1, help="client_sample_rate")
    parser.add_argument('--decay_factor', type=float, default=0.97, help="decay factor of learning rate")
    parser.add_argument('--aggr_method', type=str, default='avg', help='aggregation method')
    parser.add_argument('--fed_dict', type=str, default='all', help='only keys in this will use fed-learning')
    parser.add_argument('--sample_method', type=str, default='iid', help="method for sampling")
    parser.add_argument('--alpha', type=float, default=0.8, help="alpha for dirichlet distribution")

    # model
    parser.add_argument('--model', type=str, default='moco', help='model name')
    parser.add_argument('--input_channel', type=int, default=3, help='input channel')
    parser.add_argument('--class_number', type=int, default=10, help='class channel')
    parser.add_argument('--use_global_queue', type=bool, default=True, help='global pool')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10_raw', help="name of dataset")
    parser.add_argument('--data_path', type=str, default='./data', help='data path')
    parser.add_argument('--resize', type=int, default=-1, help='resize the input image, -1 means no resizing')

    args = parser.parse_args()
    return args
