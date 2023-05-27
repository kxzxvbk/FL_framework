# Test Residual
python run.py --logging_path ./logging/cifar100_res_normal_avg_iid --model testres_normal --class_number 100 --dataset cifar100
python run.py --logging_path ./logging/cifar100_res_normal_cent --model testres_normal --client_num 1 --class_number 100 --dataset cifar100
# Test CNN
python run.py --logging_path ./logging/cifar100_cnn_normal_avg_iid --model testcnn_normal --class_number 100 --dataset cifar100
python run.py --logging_path ./logging/cifar100_cnn_normal_cent --model testcnn_normal --client_num 1 --class_number 100 --dataset cifar100

