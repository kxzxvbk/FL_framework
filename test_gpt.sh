python run_nlp.py --logging_path ./logging/shakespear_gpt_layer2_cent --model gpt --n_layer 2 --client_num 1
python run_nlp.py --logging_path ./logging/shakespear_gpt_layer4_cent --model gpt --n_layer 4 --client_num 1
python run_nlp.py --logging_path ./logging/shakespear_gpt_layer6_cent --model gpt --n_layer 6 --client_num 1
python run_nlp.py --logging_path ./logging/shakespear_gpt_layer8_cent --model gpt --n_layer 8 --client_num 1

python run_nlp.py --logging_path ./logging/shakespear_gpt_layer2_avg_iid --model gpt --n_layer 2
python run_nlp.py --logging_path ./logging/shakespear_gpt_layer4_avg_iid --model gpt --n_layer 4
python run_nlp.py --logging_path ./logging/shakespear_gpt_layer6_avg_iid --model gpt --n_layer 6
python run_nlp.py --logging_path ./logging/shakespear_gpt_layer8_avg_iid --model gpt --n_layer 8

