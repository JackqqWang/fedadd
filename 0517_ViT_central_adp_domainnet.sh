# centralized zero-shot

CUDA_VISIBLE_DEVICES=0 python Centralized_training.py --dataset domainnet --model ViT_S --mode train_syn_real | tee results/0517_vit_tiny_central_domainnet_train_one_adp.txt
