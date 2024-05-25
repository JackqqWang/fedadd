# centralized one adapter

CUDA_VISIBLE_DEVICES=0 python Centralized_training.py --dataset domainnet --centralized_epochs 5 --model RN50 --mode train_syn_real | tee results/0517_RN_50_central_domainnet_adp.txt
