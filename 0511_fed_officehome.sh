# officehome
CUDA_VISIBLE_DEVICES=1 python FedAdapter_debug.py --dataset officehome --interact_epoch 2 --lr_local 1e-4 --num_clients 3 --out_domain 1 | tee results/0511_fed_officehome.txt
