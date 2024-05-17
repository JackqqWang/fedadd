# officehome
CUDA_VISIBLE_DEVICES=1 python FedAdapter.py --dataset officehome --server_syn_size 750 --interact_epoch 10 --lr 1e-3 --lr_local 1e-4 --num_clients 3 --out_domain 1 | tee results/0511_fed_officehome_syn_750.txt
CUDA_VISIBLE_DEVICES=1 python FedAdapter.py --dataset officehome --server_syn_size 500 --interact_epoch 10 --lr 1e-3 --lr_local 1e-4 --num_clients 3 --out_domain 1 | tee results/0511_fed_officehome_syn_500.txt
CUDA_VISIBLE_DEVICES=1 python FedAdapter.py --dataset officehome --server_syn_size 250 --interact_epoch 10 --lr 1e-3 --lr_local 1e-4 --num_clients 3 --out_domain 1 | tee results/0511_fed_officehome_syn_250.txt
