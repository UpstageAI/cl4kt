# Contrastive Learning for Knowledge Tracing
This is our implementation for the paper "Contrastive Learning for Knowledge Tracing" (TheWebConf 2022).

To run CL4KT, please create the configuration file (`/configs/example.yaml`) and the raw dataset (e.g., `/datatset/algebra05/data.txt`, `/datatset/assistments09/data.csv`, etc.).

Please use the following script to run data preprocessing:
`python preprocess_data.py --data_name algebra05 --min_user_inter_num 5`

Please use the following script to run the CL4KT model:

`CUDA_VISIBLE_DEVICES=0 python main.py --model_name cl4kt --data_name algebra05 --mask_prob 0.5 --crop_prob 0.3 --permute_prob 0.5 --replace_prob 0.5 --reg_cl 0.1`
