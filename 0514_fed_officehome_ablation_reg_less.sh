# officehome
CUDA_VISIBLE_DEVICES=1 python FedAdapter_debug.py --dataset officehome --interact_epoch 2 --lam 0 --lr 1e-3 --lr_local 1e-4 --num_clients 3 --epoch 3 --out_domain 1 | tee results/0514_fed_officehome_reg_less.txt
# CUDA_VISIBLE_DEVICES=1 python FedAdapter_debug.py --dataset officehome --interact_epoch 2 --lr 1e-3 --lr_local 1e-4 --num_clients 3 --out_domain 1 | tee results/0514_fed_officehome_500.txt
# CUDA_VISIBLE_DEVICES=1 python FedAdapter_debug.py --dataset officehome --interact_epoch 2 --lr 1e-3 --lr_local 1e-4 --num_clients 3 --out_domain 1 | tee results/0514_fed_officehome_250.txt

