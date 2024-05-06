import argparse
import torch
def arg_parser():
    
    parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser(description='CLIP on OfficeHome dataset')
    parser.add_argument('--model', default='ViT_B_32', type=str, help='pre_trained CLIP')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--category', default='Art', type=str, help='category of OfficeHome dataset')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--communication_rounds', default=50, type=int, help='number of communication rounds')
    parser.add_argument('--epoch', default=5, type=int, help='number of local training epochs')
    parser.add_argument('--adp_init_epoch', default=20, type=int, help='number of adapter initialization epochs')
    parser.add_argument('--mul_KD', action='store_true', help='mutual KD')
    parser.add_argument('--lr_CLIP', default=1e-4, type=float, help='learning rate for server CLIP')
    parser.add_argument('--lr_local', default=3e-4, type=float, help='learning rate for local classifier')
    parser.add_argument('--lr_KD', default=1e-4, type=float, help='learning rate for KD')
    parser.add_argument('--temperature', default=1, type=float, help='temperature for KD')
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha (weight for soft target loss) for KD')
    parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--adp', action='store_true', help='only train adapter')
    parser.add_argument('--server_ratio', default=0.3, type=float, help='server data ratio')
    parser.add_argument('--test_ratio', default=0.2, type=float, help='test data ratio')

    # parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    print(args)
    
    return args