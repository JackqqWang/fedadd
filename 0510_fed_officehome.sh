# officehome
CUDA_VISIBLE_DEVICES=1 python FedAdapter.py --dataset officehome --interact_epoch 10 --lr 1e-3 --lr_local 1e-4 --num_clients 3 --out_domain 1 | tee results/0510_fed_officehome.txt
