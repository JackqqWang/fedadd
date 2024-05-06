import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import os
from models.CLIP_adapter import ClipModelatFed
from tqdm import tqdm
import copy as cp
from utils_FedAdapter import arg_parser, Split_office_home_dataset, distribute_local_data
from utils_FedAdapter import knowledge_distillation_loss
from utils_FedAdapter import set_seed
from utils_FedAdapter import centralized_train, centralized_zero_shot
from Client import Client
from options_fedadapter import *

dataset_to_categories = {'officehome': ['Art', 'Clipart', 'Product', 'Real_World'],
                         'domainnet': ['clipart', 'infograph', 'painting', 'quickdraw', 'realworld', 'sketch'],
                         'imageclef_da': ['c', 'i', 'p']}

def main():

    args = arg_parser()
    device = args.device

    set_seed(args.seed)

    assert args.num_client_cat==1, "each client should have only one domain"
    assert args.num_clients % len(dataset_to_categories[args.dataset]) == 0, "number of clients should be multiple of number of domains"

    categories = dataset_to_categories[args.dataset]
    
    # init CLIP 
    CLIP_adapter = ClipModelatFed(args, adp=True)

    device = args.device

    # load and split data
    print('split **real** data for server and clients')
    server_data, local_data, classes = Split_office_home_dataset(args, CLIP_adapter.clip_preprocess, categories)

    # insert data to server and create adapters
    print('create adpaters for each domain and insert server data')
    for i in range(len(categories)):
        CLIP_adapter.insert_datasets(server_data[i], classes[i], categories[i])
        CLIP_adapter.init_adapter(i)
    
    # distribute local data to clients
    print('distribute local data to clients')
    client_datasets, category_to_clients, client_to_categories = distribute_local_data(local_data, args.num_clients, len(categories), args)
    print(f'category_to_clients: {category_to_clients}')
    print(f'client_to_categories: {client_to_categories}')

    # init clients
    print('create clients'), 
    clients = []
    for i in range(args.num_clients):
        num_classes = 0
        for domain in client_to_categories[i]:
            num_classes += len(classes[domain])
        clients.append(Client(client_datasets[i], num_classes, args))

    # server side:
    # 0.1 initialize adapters for each category (based on server data)
    print("Initialize adapters for each category")
    for i in range(len(categories)):
        CLIP_adapter.train_adapter(i, categories[i], args.adp_init_epoch, init=True)

    # 0.2 initialize weights for adapters
    CLIP_adapter.init_adapters_weights()

    # server-client communication
    for epoch in range(args.communication_rounds):

        # client side:
        # 1.train local image classifier (based on local data)
        print("-------Communication round {}--------".format(epoch))
        print("Start local training...")

        for i in range(len(clients)):
            
            print(f"Train local client {i}:")
            clients[i].train_classifier()


        print("Finish local training.")

        print("Start server update...")

        # server side:
        # 2.mutual KD
        print("Start Mutual KD:")  
        for mul_e in range(args.KD_epoch):  
            print(f"Mutual KD epoch {mul_e}") 
            
            # 2.1 adapter is student

            print("Adapter is student, update adapters...")    
            # for each adapter i
            for i in range(len(categories)):
                optimizer = torch.optim.Adam(CLIP_adapter.adapters[i].parameters(), lr=args.lr_KD)

                # select all clients has the same category
                clients_with_domain_i = category_to_clients[i]
                image_classifiers_with_domain_i = [cp.deepcopy(clients[c].image_classifier) for c in clients_with_domain_i]

                # use server data for KD
                train_loader = DataLoader(CLIP_adapter.trainsets[i], batch_size=args.batch_size, shuffle=True)

                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    # teacher_logits = avg(all clients with domain i logits)
                    teacher_logits = torch.zeros((images.shape[0], len(CLIP_adapter.class_names[i]))).to('cuda')
                    for img_classifer in image_classifiers_with_domain_i:
                        img_classifer.eval()
                        with torch.no_grad():
                            outputs = img_classifer(images)
                            teacher_logits += outputs

                    teacher_logits = teacher_logits/len(clients_with_domain_i)

                    student_logits = CLIP_adapter.CLIP_logtis(i, True, images)

                    loss = knowledge_distillation_loss(student_logits, teacher_logits, labels, args.temperature, args.alpha)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
            # 2.2 adapter is teacher
            print("Adapter is teacher, update clients...")        
            # for each client
            for j, client in enumerate(clients):
                domains_of_client = client_to_categories[j]

                # now assume each client has only one domain
                assert len(domains_of_client) == 1, "client should have only one domain"
                domain = domains_of_client[0]

                optimizer = torch.optim.Adam(client.image_classifier.parameters(), lr=args.lr_KD)

                # use server data for KD
                train_loader = DataLoader(CLIP_adapter.trainsets[domain], batch_size=args.batch_size, shuffle=True)

                client.image_classifier.train()

                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    student_logits = client.image_classifier(images)

                    with torch.no_grad():
                        teacher_logits = CLIP_adapter.CLIP_logtis(domain, False, images)

                    loss = knowledge_distillation_loss(student_logits, teacher_logits, labels, args.temperature, args.alpha)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # server side:
        # train adapters for each category (based on server data)
        print(f"Fine tune adapters after FL for each category for {args.server_FT} epochs...")
        for i in range(len(categories)):
            CLIP_adapter.train_adapter(i, categories[i], args.server_FT)
        
        # adapeters' performance
        print("Start adapter performance evaluation... (before weighted adpater fusion)")
        CLIP_adapter.inference_all_adapters(epoch)

        # 3. update weights for adapters (use server data)
        for _ in range(args.interact_epoch):
            # switch is False, update weights and fix adapters
            CLIP_adapter.train_adapters_and_weights(False)
            # switch is True, update adapters and fix weights
            CLIP_adapter.train_adapters_and_weights(True)
            
        print("Finish server update.")

        # adapeters' performance
        print("Start adapter performance evaluation... (after weighted adpater fusion)")
        CLIP_adapter.inference_all_adapters(epoch)
        CLIP_adapter.inference_weighted_adapters(epoch)

    
    print("--------Finish Communication round {}--------".format(epoch))

    for i, (acc, r) in enumerate(CLIP_adapter.best_acc):
        print(f"Category {i} best accuracy: {acc} at round {r}")


if __name__ == "__main__":
    main()
