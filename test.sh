python run.py --logging_path ./logging/imagenet_tiny_resnet9_avg_iid --model resnet9
python run.py --logging_path ./logging/imagenet_tiny_resnet18_avg_iid --model resnet18
python run.py --logging_path ./logging/imagenet_tiny_resnet34_avg_iid --model resnet34
python run.py --logging_path ./logging/imagenet_tiny_resnet50_avg_iid --model resnet50

python run.py --logging_path ./logging/imagenet_tiny_resnet9_cent --model resnet9 --client_num 1
python run.py --logging_path ./logging/imagenet_tiny_resnet18_cent --model resnet18 --client_num 1
python run.py --logging_path ./logging/imagenet_tiny_resnet34_cent --model resnet34 --client_num 1
python run.py --logging_path ./logging/imagenet_tiny_resnet50_cent --model resnet50 --client_num 1