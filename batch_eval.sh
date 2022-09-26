#!/bin/bash
path_prefix=./logging/cifar10_resnet18_dirichlet_fedavg_alpha_

for ((i=2;i<=8;i+=2))
do
        python run.py --logging_path $path_prefix0.$i --alpha 0.$i
done