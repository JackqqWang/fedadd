# domainnet
CUDA_VISIBLE_DEVICES=3 python FedAdapter_debug.py --dataset domainnet --num_clients 3 --server_ratio 50 --interact_epoch 2 --out_domain 1,3,5 --model ViT_S --server_syn_size 20000 | tee results/0517_vit_tiny_fed_domainnet.txt
