import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from models.CLIP_adapter import ClipModelatFed
from models.CLIP_one_adapter import ClipModelatFed as ClipModelatFed_one
from tqdm import tqdm
import copy as cp
from utils_FedAdapter import arg_parser, Split_office_home_dataset, distribute_local_data
from utils_FedAdapter import knowledge_distillation_loss
from utils_FedAdapter import set_seed
from utils_FedAdapter import centralized_train, centralized_zero_shot
from options_fedadapter import *

dataset_to_categories = {'officehome': ['Art', 'Clipart', 'Product', 'Real_World'],
                         'domainnet': ['clipart', 'infograph', 'painting', 'quickdraw', 'realworld', 'sketch'],
                         'imageclef_da': ['c', 'i', 'p']}


# def main_old():

#     # each domain has one adapter

#     args = arg_parser()

#     set_seed(args.seed)

#     categories = dataset_to_categories[args.dataset]

#     # init CLIP 
#     CLIP_adapter = ClipModelatFed(args, adp=True)

#     # load and split data
#     print('split **real** data for server and clients')
#     server_data, local_data, classes = Split_office_home_dataset(args, CLIP_adapter.clip_preprocess, categories)

#     # insert data to server and create adapters
#     print('create adpaters for each domain and insert server data')
#     for i in range(len(categories)):
#         CLIP_adapter.insert_datasets(server_data[i], classes[i], categories[i])
#         CLIP_adapter.init_adapter(i)

#     if args.mode == 'zero_shot':
#         print('zero shot')
#         for i in range(len(categories)):
#             centralized_zero_shot(CLIP_adapter, i, args)
        
#     else:
#         print(f'train on {args.mode}')
#         for i in range(len(categories)):
#             centralized_train(CLIP_adapter, local_data, i, categories[i], args)


def main():

    # one adapter for all domains

    args = arg_parser()

    set_seed(args.seed)

    # categories = dataset_to_categories[args.dataset]
    # in-domain -> training, out-domain ->testing
    out_domain = [int(e) for e in args.out_domain.split(',')]
    categories = [domain for ind, domain in enumerate(dataset_to_categories[args.dataset]) if ind not in out_domain]
    test_categories = [domain for ind, domain in enumerate(dataset_to_categories[args.dataset]) if ind in out_domain]

    print(f"in-domain: {categories}")
    print(f"out-domain: {test_categories}")

    # init CLIP 
    CLIP_adapter = ClipModelatFed_one(args, adp=True)

    # load and split data (out-domains)
    print('split **real** data for server (out-domains)')
    server_data, _, classes = Split_office_home_dataset(args, CLIP_adapter.clip_preprocess, test_categories)
    for i in range(len(test_categories)):
        CLIP_adapter.insert_out_testsets(server_data[i], classes[i])

    # load and split data (in-domains)
    print('split **real** data for server and clients (in-domains)')
    server_data, local_data, classes = Split_office_home_dataset(args, CLIP_adapter.clip_preprocess, categories)

    # insert data to server and create adapters (in-domains)
    print('insert server data for each domain')
    for i in range(len(categories)):
        CLIP_adapter.insert_datasets(server_data[i], classes[i], categories[i])
    
    CLIP_adapter.init_adapter()
        
    if args.mode == 'zero_shot':
        print('zero shot')
        for i in range(len(categories)):
            centralized_zero_shot(CLIP_adapter, i, args)
        
    else:
        print(f'train on {args.mode}')
        CLIP_adapter.train_adapter(local_data, categories, test_categories)


if __name__ == "__main__":
    main()
