import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import os
from models.Baseline_Fed_server import Server_ViT, Client_ViT
from tqdm import tqdm
import clip
import copy as cp
from utils_FedAdapter import arg_parser, Split_office_home_dataset, distribute_local_data
from utils_FedAdapter import knowledge_distillation_loss
from utils_FedAdapter import set_seed
from options_fedadapter import *

dataset_to_categories = {'officehome': ['Art', 'Clipart', 'Product', 'Real_World'],
                         'domainnet': ['clipart', 'infograph', 'painting', 'quickdraw', 'realworld', 'sketch'],
                         'imageclef_da': ['c', 'i', 'p']}

def main():
    
    args = arg_parser()
    print(args.mode)
    set_seed(args.seed)

    # categories = dataset_to_categories[args.dataset]
    # in-domain -> training, out-domain ->testing
    out_domain = [int(e) for e in args.out_domain.split(',')]
    categories = [domain for ind, domain in enumerate(dataset_to_categories[args.dataset]) if ind not in out_domain]
    test_categories = [domain for ind, domain in enumerate(dataset_to_categories[args.dataset]) if ind in out_domain]

    print(f"in-domain: {categories}")
    print(f"out-domain: {test_categories}")

    assert args.num_client_cat==1, "each client should have only one domain"
    assert args.num_clients % len(categories) == 0, "number of clients should be multiple of number of in-domains"

    _, clip_preprocess = clip.load('ViT-B/16', device=args.device, jit=False)
    
    # load and split data (out-domains)
    print('split **real** data for server (out-domains)')
    server_data, _, classes = Split_office_home_dataset(args, clip_preprocess, test_categories)

    # create server
    server = Server_ViT(args, len(classes[0]), clip_preprocess)

    for i in range(len(test_categories)):
        server.insert_out_testsets(server_data[i], classes[i])

    # load and split data (in-domains)
    print('split **real** data for server and clients (in-domains)')
    server_data, local_data, classes = Split_office_home_dataset(args, clip_preprocess, categories)

    # insert data to server (in-domains)
    print('create adpaters for each domain and insert server data')
    for i in range(len(categories)):
        server.insert_datasets(server_data[i], classes[i], categories[i])

    for epoch in range(args.centralized_epochs):
  
        print(f'centralized training epoch {epoch}...')

        if 'real' in args.mode:
            trainsets = ConcatDataset([trainset for trainset in server.trainsets] + [dataset for dataset in local_data])
        else:
            trainsets = ConcatDataset([trainset for trainset in server.trainsets])
        
        train_loader = DataLoader(trainsets, batch_size=args.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(server.image_classifier.parameters(), lr=args.centralized_lr)

        server.train(optimizer, train_loader)

        # indomain test
        for i in range(len(server.testsets)):
            test_loader = DataLoader(server.testsets[i], batch_size=args.batch_size, shuffle=False)
            test_acc_i = server.test(test_loader)
            print(f"in-domain {categories[i]} has accuracy: {test_acc_i}")
        # outdomain test
        for i in range(len(server.testsets_out)):
            test_loader = DataLoader(server.testsets_out[i], batch_size=args.batch_size, shuffle=False)
            accuracy = server.test(test_loader)
            print(f"out-domain {test_categories[i]} has accuracy: {accuracy}")


if __name__ == '__main__':
    main()
