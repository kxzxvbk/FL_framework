from matplotlib import pyplot as plt


def plot_data(data_dic):
    for k in data_dic:
        plt.plot(data_dic[k][0], data_dic[k][1], label=k)
    plt.legend(ncol=len(data_dic))
    plt.show()


def load_from_log(path):
    def _get_dict_from_line(lin):
        lin = lin.split('    ')[1]
        lin = lin.split(',')
        ret_dict = {}
        for term in lin:
            term = term.split(':')
            term[0], term[1] = term[0].strip(), term[1].strip()
            if term[1][-1] == 'M':
                term[1] = term[1][:-1]
            try:
                k, v = term[0], eval(term[1])
            except:
                print(term)
                print(path)
                assert False
            ret_dict[k] = v
        if 'trans_cost' not in ret_dict:
            ret_dict['trans_cost'] = 335.7379

        return ret_dict

    ret_list = []
    with open(path) as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while len(line) != 0:
            dict_train = _get_dict_from_line(line)
            line = f.readline()
            dict_test = _get_dict_from_line(line)
            ret_list.append((dict_train, dict_test))
            line = f.readline()
    return ret_list


def plot_two_keys(k1, k2, name2file):
    def _acc(x):
        tmp_sum = 0
        for i in range(len(x)):
            tmp_sum += x[i]
            x[i] = tmp_sum

    def _add_acc(x):
        tmp_sum = 0
        for i in range(len(x)):
            tmp_sum += x[i]
            if i % 10 == 0:
                tmp_sum += 335.7379
            x[i] = tmp_sum

    assert k1 in ['round', 'trans_cost']
    assert k2 in ['train_acc', 'test_acc', 'train_loss', 'test_loss']

    res_dict = {}

    for key in name2file:
        try:
            dic = load_from_log(name2file[key])
            if k1 == 'round':
                x = range(len(dic))
            else:
                x = [dic[i][0]['trans_cost'] for i in range(len(dic))]
                if 'fedcomp' in key:
                    _add_acc(x)
                else:
                    _acc(x)
            if 'train' in k2:
                y = [dic[i][0][k2] for i in range(len(dic))]
            else:
                y = [dic[i][1][k2] for i in range(len(dic))]
            res_dict[key] = (x, y)
        except:
            print(key)
            assert False

    plot_data(res_dict)


if __name__ == '__main__':
    plot_two_keys('trans_cost', 'train_acc', {
        'fed_avg': '../logging/fedavg_cifar10_resnet18',
        # 'FedComp_EC': '../logging/FedComp_cifar10_resnet18_interval=1_ec',
        # 'FedComp-interval=1': '../logging/FedComp_cifar10_resnet18_interval=1_no_ec',
        # 'top-k-10': '../logging/topk10_cifar10_resnet18',
        # 'top-k': '../logging/topk_20_cifar10_resnet18',
        # 'qsgd': '../logging/qsgd_cifar10_resnet18',
        # 'F3': '../logging/FedComp_cifar10_resnet18_interval=3_no_ec',
        # 'F3ec': '../logging/FedComp_cifar10_resnet18_interval=3_new_ec',
        # 'F3ec—new': '../logging/FedComp_cifar10_resnet18_interval=3_new_new_ec',
        # 'F3-rescale=2': '../logging/FedComp_cifar10_resnet18_interval=3_rescale2',
        # 'F3-autoscale': '../logging/FedComp_cifar10_resnet18_interval=3_autoscale',
        # 'F3-sqrt-autoscale': '../logging/FedComp_cifar10_resnet18_interval=3_sqrt_autoscale',
        # 'F3-clip-autoscale': '../logging/FedComp_cifar10_resnet18_interval=3_clip_autoscale',
        # 'F3-clip-bias': '../logging/FedComp_cifar10_resnet18_interval=3_clip_bias_comp',
        # 'F3-clip-except-bn': '../logging/FedComp_cifar10_resnet18_interval=1_clip33_autoscale_except_bn0',
        # 'powersgd': '../logging/powersgd10_cifar10_resnet18',
        # 'fedcomp': '../logging/fedcomp_cifar10_resnet18_interval=10_sota0',
        'fedcomp-rescale-PID': '../logging/resnet18_cifar10_integration',
        # 'qsgd': '../logging/resnet18_cifar10_qsgd',
        # 'fedcomp+rescale': '../logging/fedcomp_cifar10_resnet18_interval=3_autoscale'
    })
