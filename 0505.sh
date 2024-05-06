# officehome
# centralized zero-shot
# CUDA_VISIBLE_DEVICES=0 python Centralized_training.py --dataset domainnet --mode zero_shot | tee results/central_domainnet_zero_shot.txt
# CUDA_VISIBLE_DEVICES=0 python Centralized_training.py --dataset imageclef_da --mode zero_shot | tee results/central_imageclef_da_zero_shot.txt


# centralized training on syn data
# CUDA_VISIBLE_DEVICES=0 python Centralized_training.py --dataset domainnet --mode train_syn --centralized_epochs 20 --centralized_lr 1e-4  | tee results/central_domainnet_train_syn.txt
# CUDA_VISIBLE_DEVICES=0 python Centralized_training.py --dataset imageclef_da --mode train_syn --centralized_epochs 20 --centralized_lr 1e-4  | tee results/central_imageclef_da_train_syn.txt

# # centralized training on syn + real data
# CUDA_VISIBLE_DEVICES=0 python Centralized_training.py --dataset officehome --mode train_syn_real --centralized_epochs 20 --centralized_lr 1e-4 | tee results/central_officehome_train_syn_real.txt
CUDA_VISIBLE_DEVICES=0 python Centralized_training.py --dataset domainnet --mode train_syn_real --centralized_epochs 20 --centralized_lr 1e-4 | tee results/central_domainnet_train_syn_real.txt
CUDA_VISIBLE_DEVICES=0 python Centralized_training.py --dataset imageclef_da --mode train_syn_real --centralized_epochs 20 --centralized_lr 1e-4 | tee results/central_imageclef_da_train_syn_real.txt


# test git push