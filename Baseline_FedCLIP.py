import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import os
from models.Baseline_Fed_server import Server_CLIP, Client_CLIP
from tqdm import tqdm
import copy as cp
from utils_FedAdapter import arg_parser, Split_office_home_dataset, distribute_local_data
from utils_FedAdapter import knowledge_distillation_loss
from utils_FedAdapter import set_seed
from options_fedadapter import *

dataset_to_categories = {'officehome': ['Art', 'Clipart', 'Product', 'Real_World'],
                         'domainnet': ['clipart', 'infograph', 'painting', 'quickdraw', 'realworld', 'sketch'],
                         'imageclef_da': ['c', 'i', 'p']}

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

    assert args.num_client_cat==1, "each client should have only one domain"
    assert args.num_clients % len(categories) == 0, "number of clients should be multiple of number of in-domains"

    # create server
    server = Server_CLIP(args)

    # load and split data (out-domains)
    print('split **real** data for server (out-domains)')
    server_data, _, classes = Split_office_home_dataset(args, server.clip_preprocess, test_categories)
    for i in range(len(test_categories)):
        server.insert_out_testsets(server_data[i], classes[i])

    # load and split data (in-domains)
    print('split **real** data for server and clients (in-domains)')
    server_data, local_data, classes = Split_office_home_dataset(args, server.clip_preprocess, categories)

    # insert data to server and create adapters (in-domains)
    print('create adpaters for each domain and insert server data')
    for i in range(len(categories)):
        server.insert_datasets(server_data[i], classes[i], categories[i])
    server.init_adapter()    

    # distribute local data to clients (in-domains)
    print('distribute local data to clients')
    client_datasets, category_to_clients, client_to_categories = distribute_local_data(local_data, args.num_clients, len(categories), args)
    print(f'category_to_clients: {category_to_clients}')
    print(f'client_to_categories: {client_to_categories}')

    # create clients
    clients = []
    for i in range(args.num_clients):
        clients.append(Client_CLIP(client_datasets[i], classes[i], args))

    for epoch in range(args.communication_rounds):
        print("-------Communication round {}--------".format(epoch))

        print("Start local training...")
        for i in range(len(clients)):
            print(f"Train local client {i}:")
            clients[i].train_adapter()
        
        print('FedAvg...')
        global_adapter = cp.deepcopy(clients[0].adapter)
        # Accumulate parameters from all other clients
        for client in clients[1:]:  # start from the second client
            for (param, client_param) in zip(global_adapter.parameters(), client.adapter.parameters()):
                param.data += client_param.data

        # Average the parameters
        num_clients = len(clients)
        for param in global_adapter.parameters():
            param.data /= num_clients

        # Assign the averaged adapter to the server
        server.adapter = global_adapter

        print('server fine-tuning...')
        server.train_adapter()

        # indomain test
        for i in range(len(server.testsets)):
            test_loader = DataLoader(server.testsets[i], batch_size=args.batch_size, shuffle=False)
            test_acc_i = server.test(server.adapter, test_loader, classes[0])
            print(f"in-domain {categories[i]} has accuracy: {test_acc_i}")
        # outdomain test
        for i in range(len(server.testsets_out)):
            test_loader = DataLoader(server.testsets_out[i], batch_size=args.batch_size, shuffle=False)
            accuracy = server.test(server.adapter, test_loader, classes[0])
            print(f"out-domain {test_categories[i]} has accuracy: {accuracy}")

        print('distribute adapter to clients')
        for client in clients:
            client.adapter = cp.deepcopy(server.adapter)


if __name__ == '__main__':
    main()
