# centralized zero-shot

CUDA_VISIBLE_DEVICES=1 python Centralized_training.py --dataset domainnet --mode zero_shot | tee results/central_domainnet_zero_shot.txt

CUDA_VISIBLE_DEVICES=1 python Centralized_training.py --dataset imageclef_da --mode zero_shot | tee results/central_imageclef_da_zero_shot.txt