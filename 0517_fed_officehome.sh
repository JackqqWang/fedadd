# officehome
CUDA_VISIBLE_DEVICES=2 python FedAdapter_debug.py --dataset officehome --interact_epoch 2 --model ViT_S --lr_local 1e-4 --num_clients 3 --out_domain 1 | tee results/0517_ViT_S_fed_officehome.txt
