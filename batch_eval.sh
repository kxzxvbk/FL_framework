#!/bin/bash
path_prefix=./logging/cifar10_resnet18_dirichlet_fedavg_alpha_0.

for ((i=2;i<=8;i+=2))
do
        python run.py --logging_path $path_prefix$i --alpha 0.$i
done