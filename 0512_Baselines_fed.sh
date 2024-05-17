# # Centralized ViT-tiny
# python Baseline_CentralViT.py --dataset officehome --out_domain 1 --num_clients 3 --centralized_epochs 10 --centralized_lr 1e-4

# Fed Vit-tiny
# CUDA_VISIBLE_DEVICES=1 python Baseline_FedViT.py --dataset officehome --out_domain 1 --num_clients 3 --epoch 5 --interact_epoch 2 --lr_CLIP 1e-4 --lr_local 1e-4 --communication_rounds 10 | tee results/0512_fed_vit_officehome.txt
# CUDA_VISIBLE_DEVICES=1 python Baseline_FedViT.py --dataset imageclef_da --out_domain 2 --num_clients 2 --epoch 5 --interact_epoch 2 --lr_CLIP 1e-4 --lr_local 1e-4 --communication_rounds 10 | tee results/0512_fed_vit_imageclef_da.txt
# CUDA_VISIBLE_DEVICES=1 python Baseline_FedViT.py --dataset domainnet --out_domain 1,3,5 --num_clients 3 --epoch 5 --interact_epoch 2 --lr_CLIP 1e-4 --lr_local 1e-4 --communication_rounds 10 | tee results/0512_fed_vit_domainnet.txt

# Fed CLIP (ViT-tiny)
# CUDA_VISIBLE_DEVICES=1 python Baseline_FedCLIP.py --dataset officehome --out_domain 1 --num_clients 3 --epoch 5 --interact_epoch 2 --lr_CLIP 1e-4 --lr_local 1e-4 --communication_rounds 10 | tee results/0512_fed_clip_officehome.txt
CUDA_VISIBLE_DEVICES=0 python Baseline_FedCLIP.py --dataset imageclef_da --out_domain 2 --num_clients 2 --epoch 5 --interact_epoch 2 --lr_CLIP 1e-4 --lr_local 1e-4 --communication_rounds 5 | tee results/0513_fed_clip_imageclef_da.txt
CUDA_VISIBLE_DEVICES=0 python Baseline_FedCLIP.py --dataset domainnet --out_domain 1,3,5 --num_clients 3 --epoch 5 --interact_epoch 2 --lr_CLIP 1e-4 --lr_local 1e-4 --communication_rounds 10 | tee results/0513_fed_clip_domainnet.txt

