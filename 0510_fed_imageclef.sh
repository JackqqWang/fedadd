# imageclef_da
CUDA_VISIBLE_DEVICES=0 python FedAdapter.py --dataset imageclef_da --num_clients 2 --lr_local 1e-4 --out_domain 2 | tee results/0510_fed_imageclef.txt
