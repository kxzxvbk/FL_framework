python run.py --logging_path ./logging/imagenet_tiny_cnn_mean_avg_iid --model testcnn_mean
python run.py --logging_path ./logging/imagenet_tiny_cnn_normal_avg_iid --model testcnn_normal
python run.py --logging_path ./logging/imagenet_tiny_cnn_anti_avg_iid --model testcnn_anti
python run.py --logging_path ./logging/imagenet_tiny_cnn_nobn_avg_iid --model testcnn_nobn

python run.py --logging_path ./logging/imagenet_tiny_cnn_mean_cent --model testcnn_mean --client_num 1
python run.py --logging_path ./logging/imagenet_tiny_cnn_normal_cent --model testcnn_normal --client_num 1
python run.py --logging_path ./logging/imagenet_tiny_cnn_anti_cent --model testcnn_anti --client_num 1
python run.py --logging_path ./logging/imagenet_tiny_cnn_nobn_cent --model testcnn_nobn --client_num 1
