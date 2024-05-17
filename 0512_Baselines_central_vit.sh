# Centralized ViT-tiny
CUDA_VISIBLE_DEVICES=1 python Baseline_CentralViT.py --dataset officehome --out_domain 1 --num_clients 3 --centralized_epochs 10 --centralized_lr 1e-4 | tee results/0512_central_vit_officehome.txt
CUDA_VISIBLE_DEVICES=1 python Baseline_CentralViT.py --dataset imageclef_da --out_domain 2 --num_clients 2 --centralized_epochs 10 --centralized_lr 1e-4 | tee results/0512_central_vit_imageclef.txt
CUDA_VISIBLE_DEVICES=1 python Baseline_CentralViT.py --dataset domainnet --out_domain 1,3,5 --num_clients 3 --centralized_epochs 10 --centralized_lr 1e-4 | tee results/0512_central_vit_domainnet.txt

# # Fed Vit-tiny
# python Baseline_FedViT.py --dataset officehome --out_domain 1 --num_clients 3 --epoch 5 --interact_epoch 2 --lr_CLIP 1e-4 --lr_local 1e-4 --communication_rounds 10

# # Fed CLIP (ViT-tiny)
# python Baseline_FedCLIP.py --dataset officehome --out_domain 1 --num_clients 3 --epoch 5 --interact_epoch 2 --lr_CLIP 1e-4 --lr_local 1e-4 --communication_rounds 10

