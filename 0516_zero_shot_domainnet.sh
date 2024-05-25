# centralized zero-shot

CUDA_VISIBLE_DEVICES=0 python Centralized_training.py --dataset domainnet --model RN50 --mode zero_shot | tee results/0517_RN_50_central_domainnet_zero_shot.txt
