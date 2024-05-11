# domainnet
CUDA_VISIBLE_DEVICES=2 python FedAdapter.py --dataset domainnet --interact_epoch 10 --lr 1e-3 --num_clients 3 --out_domain 1,3,5 --server_syn_size 20000 | tee results/0510_fed_domainnet.txt
