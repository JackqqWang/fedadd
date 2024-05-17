# domainnet
CUDA_VISIBLE_DEVICES=2 python FedAdapter_debug.py --dataset domainnet --num_clients 3 --interact_epoch 2 --out_domain 1,3,5 --server_syn_size 20000 | tee results/0511_fed_domainnet.txt
