python run.py --logging_path ./logging/imagenet_tiny_res_mean_avg_iid --model testres_mean
python run.py --logging_path ./logging/imagenet_tiny_res_normal_avg_iid --model testres_normal
python run.py --logging_path ./logging/imagenet_tiny_res_anti_avg_iid --model testres_anti
python run.py --logging_path ./logging/imagenet_tiny_res_nobn_avg_iid --model testres_nobn

python run.py --logging_path ./logging/imagenet_tiny_res_mean_cent --model testres_mean --client_num 1
python run.py --logging_path ./logging/imagenet_tiny_res_normal_cent --model testres_normal --client_num 1
python run.py --logging_path ./logging/imagenet_tiny_res_anti_cent --model testres_anti --client_num 1
python run.py --logging_path ./logging/imagenet_tiny_res_nobn_cent --model testres_nobn --client_num 1
