#!/bin/bash
#run experiements with different level of non-iid
path_prefix=./logging/cifar10_resnet18_dirichlet_fedavg_alpha_

for ((i=2;i<=8;i+=2))
do
        echo "python run.py --log_dir"$path_prefix"0.$i --alpha 0.$i"
done