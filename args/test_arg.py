import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--loc_eps', type=int, default=10, help="rounds of training")
    parser.add_argument('--glob_eps', type=int, default=500, help="global training round")
    parser.add_argument('--client_num', type=int, default=30, help="number of client")
    parser.add_argument('--client_sample_rate', type=float, default=0.5, help="client_sample_rate")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--resize', type=int, default=112, help='resize the input image, -1 means no resizing')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")

    # model
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--input_channel', type=int, default=3, help='input channel')
    parser.add_argument('--class_number', type=int, default=10, help='class channel')

    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--resume', type=bool, default=False, help="whether to resume")
    parser.add_argument('--start_round', type=int, default=0, help='round to start with')
    parser.add_argument('--device', type=int, default=0, help="GPU ID, -1 for CPU")

    parser.add_argument('--sample_method', type=str, default='iid', help="method for sampling")
    parser.add_argument('--test_freq', type=int, default=1, help="rounds of testing")
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help='loss type')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer type')
    parser.add_argument('--aggr_method', type=str, default='avg', help='aggregation method')
    parser.add_argument('--logging_path', type=str, default='./logging/avg_round10', help='logging path')
    parser.add_argument('--data_path', type=str, default='./data', help='data path')
    parser.add_argument('--fed_dict', type=str, default='all', help='only keys in this will use fed-learning')

    args = parser.parse_args()
    return args
