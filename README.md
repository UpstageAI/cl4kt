# Contrastive Learning for Knowledge Tracing
This is our implementation for the paper "Contrastive Learning for Knowledge Tracing" (TheWebConf 2022).

To run CL4KT, please specify the configuration file (`/configs/example.yaml`) and preprocess dataset (e.g., `/datatset/algebra05`).

Then, use the following command:

`CUDA_VISIBLE_DEVICES=0 python main.py --model_name cl4kt --data_name algebra05 --mask_prob 0.5 --crop_prob 0.3 --permute_prob 0.5 --replace_prob 0.5 --reg_cl 0.1`
