# officehome
# centralized zero-shot
# CUDA_VISIBLE_DEVICES=1 python Centralized_training.py --dataset officehome --mode zero_shot | tee results/central_officehome_zero_shot.txt

# centralized training on syn data
CUDA_VISIBLE_DEVICES=1 python Centralized_training.py --dataset officehome --mode train_syn --centralized_epochs 20 --centralized_lr 1e-4  | tee results/central_officehome_train_syn.txt

# centralized training on syn + real data
CUDA_VISIBLE_DEVICES=1 python Centralized_training.py --dataset officehome --mode train_syn_real --centralized_epochs 20 --centralized_lr 1e-4 | tee results/central_officehome_train_syn_real.txt
