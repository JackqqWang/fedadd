# Centralized ViT-tiny
python Baseline_CentralViT.py --dataset officehome --out_domain 1 --num_clients 3 --centralized_epochs 10 --centralized_lr 1e-4

# Fed Vit-tiny
python Baseline_FedViT.py --dataset officehome --out_domain 1 --num_clients 3 --epoch 5 --interact_epoch 2 --lr_CLIP 1e-4 --lr_local 1e-4 --communication_rounds 10

# Fed CLIP (ViT-tiny)
python Baseline_FedCLIP.py --dataset officehome --out_domain 1 --num_clients 3 --epoch 5 --interact_epoch 2 --lr_CLIP 1e-4 --lr_local 1e-4 --communication_rounds 10

