# centralized zero-shot

CUDA_VISIBLE_DEVICES=0 python Centralized_training.py --dataset officehome --model RN50 --mode zero_shot | tee results/0517_RN_50_central_officehome_zero_shot.txt
