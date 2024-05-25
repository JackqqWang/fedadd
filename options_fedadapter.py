
import argparse
import torch

def arg_parser():
    
    parser = argparse.ArgumentParser(description='CLIP on OfficeHome dataset')
    parser.add_argument('--model', default='ViT_B_32', type=str, help='pre_trained CLIP')
#     CLIP_MODELS = {
#     'RN50':'RN50',
#     'RN101':'RN101',
#     'RN50x4':'RN50x4',
#     'RN50x16':'RN50x16',
#     'RN50x64':'RN50x64',
#     'ViT_B_32':'ViT-B/32',
#     'ViT_B_16':'ViT-B/16',
#     'ViT_L_14':'ViT-L/14',
#     'ViT_L_14_336px':'ViT-L/14@336px'
# }
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    # parser.add_argument('--category', default='Art', type=str, help='category of OfficeHome dataset')
    parser.add_argument('--dataset', default='officehome', type=str, help='dataset')
    parser.add_argument('--data_dir', default='../../../data/jiaqi/fedadapter', type=str, help='data directory')
    parser.add_argument('--out_domain', default='1', type=str, help='out domains: e.g, 1,2,3')
    parser.add_argument('--num_clients', default=4, type=int, help='number of clients')
    parser.add_argument('--num_client_cat', default=1, type=int, help='number of categories for each client')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--communication_rounds', default=10, type=int, help='number of communication rounds')
    parser.add_argument('--epoch', default=10, type=int, help='number of local training epochs')
    parser.add_argument('--adp_init_epoch', default=20, type=int, help='number of adapter initialization epochs')
    parser.add_argument('--interact_epoch', default=2, type=int, help='number of interaction epochs')
    # parser.add_argument('--KD_epoch', default=2, type=int, help='number of KD epochs')
    # parser.add_argument('--interact_epoch', default=1, type=int, help='interact epochs')
    # parser.add_argument('--server_FT', default=0, type=int, help='number of server fine tuning epochs')
    # parser.add_argument('--mul_KD', action='store_true', help='mutual KD')
    parser.add_argument('--lr_CLIP', default=1e-4, type=float, help='learning rate for server CLIP')
    parser.add_argument('--lr_local', default=3e-4, type=float, help='learning rate for local classifier')
    # parser.add_argument('--lr_KD', default=1e-4, type=float, help='learning rate for KD')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for end2end training')
    parser.add_argument('--temperature', default=1, type=float, help='temperature for KD')
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha (weight for soft target loss) for KD')
    # parser.add_argument('--beta', default=0.1, type=float, help='beta: weight for weighted sum of adapter outputs')
    parser.add_argument('--momentum', default=0.9, type=float, help='coefficients for old client parameters')
    parser.add_argument('--lam', default=0.1, type=float, help='lambda: weight for attention regularization')
    parser.add_argument('--eps', default=1e-5, type=float, help='eps: small value for numerical stability')
    parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--server_ratio', default=0.1, type=float, help='server data ratio (only used for testing)')
    parser.add_argument('--test_ratio', default=0.2, type=float, help='test data ratio for each client')
    parser.add_argument('--server_syn_size', default=1000, type=int, help='number of synthetic server data for each domain, 0 means no synthetic data')
    parser.add_argument('--server_syn_ver', default=2.0, type=float, help='version of stable diffusion, 1 or 2')
    # parser.add_argument('--resume', action='store_true', help='resume from checkpoint')

    # only for centralized training
    parser.add_argument('--mode', default='zero_shot', type=str, help='zero_shot, train_syn, train_syn_real')
    parser.add_argument('--centralized_epochs', default=10, type=int, help='number of centralized training epochs')
    # parser.add_argument('--centralized_batch_size', default=32, type=int, help='batch size for centralized training')
    parser.add_argument('--centralized_lr', default=1e-4, type=float, help='learning rate for centralized training')

    args = parser.parse_args()
    
    return args
