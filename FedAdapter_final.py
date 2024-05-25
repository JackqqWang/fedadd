import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import pandas as pd
import os
from models.CLIP_adapter import ClipModelatFed
from tqdm import tqdm
import copy as cp
from itertools import chain
from utils_FedAdapter import arg_parser, Split_office_home_dataset, distribute_local_data
from utils_FedAdapter import mutual_KD_loss, knowledge_distillation_loss
from utils_FedAdapter import set_seed, cos_sim_tensor
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

    # categories = dataset_to_categories[args.dataset]
    # in-domain -> training, out-domain ->testing
    out_domain = [int(e) for e in args.out_domain.split(',')]
    categories = [domain for ind, domain in enumerate(dataset_to_categories[args.dataset]) if ind not in out_domain]
    test_categories = [domain for ind, domain in enumerate(dataset_to_categories[args.dataset]) if ind in out_domain]

    print(f"in-domain: {categories}")
    print(f"out-domain: {test_categories}")
    assert args.num_client_cat==1, "each client should have only one domain"
    assert args.num_clients % len(categories) == 0, "number of clients should be multiple of number of in-domains"
    
    # create CLIP on server
    CLIP_adapter = ClipModelatFed(args, adp=True)
    device = args.device

    # load and split data (out-domains)
    print('split **real** data for server (out-domains)')
    server_data, _, classes = Split_office_home_dataset(args, CLIP_adapter.clip_preprocess, test_categories)
    for i in range(len(test_categories)):
        CLIP_adapter.insert_out_testsets(server_data[i], classes[i])

    # load and split data (in-domains)
    print('split **real** data for server and clients (in-domains)')
    server_data, local_data, classes = Split_office_home_dataset(args, CLIP_adapter.clip_preprocess, categories)

    # insert data to server and create adapters (in-domains)
    print('create adpaters for each domain and insert server data')
    for i in range(len(categories)):
        CLIP_adapter.insert_datasets(server_data[i], classes[i], categories[i])
        CLIP_adapter.init_adapter()
    
    # distribute local data to clients (in-domains)
    print('distribute local data to clients')
    client_datasets, category_to_clients, client_to_categories = distribute_local_data(local_data, args.num_clients, len(categories), args)
    print(f'category_to_clients: {category_to_clients}')
    print(f'client_to_categories: {client_to_categories}')

    # create clients
    print('create clients'), 
    clients = []
    for i in range(args.num_clients):
        num_classes = 0
        for domain in client_to_categories[i]:
            num_classes += len(classes[domain])
        clients.append(Client(client_datasets[i], num_classes, args))

    # server side:
    # 0.1 initialize adapters for each domain using on server syn data (in-domains)
    print("Initialize adapters for each (in)domain")
    for i in range(len(categories)):
        CLIP_adapter.train_adapter(i, categories[i], args.adp_init_epoch, init=True)

    # 0.2 init MLP for processing logits
    CLIP_adapter.init_MLP()

    CE_loss =  torch.nn.CrossEntropyLoss()

    # FL: server-client communication
    for epoch in range(args.communication_rounds):
        

        # client side:
        # C.1 train local image classifier using local real data
        print("-------Communication round {}--------".format(epoch))
        print("Start local training...")

        for i in range(len(clients)):
            
            print(f"Train local client {i}:")
            clients[i].train_classifier()

        print("Finish local training.")

        # C.2 prototype representation calculation
        print("Calculate prototype representation (real client data)...")

        # register forward hook in each client model -> obtain embedding in pernultimate layer
        activations, hooks = {}, []
        def get_pernultimate_activation(module, input, output):
            activations['pernultimate'] = input[0]
        
        for i in range(len(clients)):
            hooks.append(clients[i].image_classifier.head.register_forward_hook(get_pernultimate_activation))

        # calculate prototype representation (real data)
        prototype_representations = {(i, y): [] for i in range(len(categories)) for y in range(len(classes[i]))}
        for i in range(len(categories)):
                
            for c in category_to_clients[i]:
                clients[c].image_classifier.eval()
                
                train_loader = DataLoader(clients[c].train_set, batch_size=args.batch_size, shuffle=False)
                
                for images, labels in train_loader:
                    images = images.to(device)
                    with torch.no_grad():
                        clients[c].image_classifier(images)
                    
                    A = activations['pernultimate'].detach().cpu()
                    for y in range(len(classes[i])):
                        # print('domain i ', i, 'class j ', j, 'size ', A[labels == j].shape)
                        prototype_representations[(i, y)].append(A[labels == y])

        prototype = []
        for i in range(len(categories)):
            prototype_i = []
            for y in range(len(classes[i])):
                prototype_representations[(i, y)] = torch.cat(prototype_representations[(i, y)], dim=0)
                prototype_representations[(i, y)] = torch.sum(prototype_representations[(i, y)], dim=0)/prototype_representations[(i, y)].shape[0]

                prototype_i.append(prototype_representations[(i, y)])
            
            prototype_i = torch.stack(prototype_i, dim=0)
            prototype.append(prototype_i)
        prototype = torch.stack(prototype, dim=0)

        # C.3 Syn Data Quality Estimation
        print("Estimate Syn Data Quality...")
        syn_data_quality = [[] for i in range(len(categories))]
        for i in range(len(categories)):
            train_loader = DataLoader(CLIP_adapter.trainsets[i], batch_size=args.batch_size, shuffle=False)
            
            for images, labels, _ in train_loader:
                images = images.to(device)
                P = prototype[i][labels]

                with torch.no_grad():
                    quality_batch = []
                    for c in category_to_clients[i]:
                        clients[c].image_classifier(images)
                        A = activations['pernultimate'].detach().cpu()
                    
                        quality_batch.append(cos_sim_tensor(A, P))
                    
                    quality_batch = torch.stack(quality_batch, dim=0)
                    quality_batch = torch.mean(quality_batch, dim=0)
                    syn_data_quality[i].append(quality_batch)
            
            syn_data_quality[i] = torch.cat(syn_data_quality[i], dim=0)
            CLIP_adapter.trainsets[i].quality_scores = syn_data_quality[i]
            
        # remove all the hooks
        for hook in hooks:
            hook.remove()


        # copy client parameters
        clients_copy = [cp.deepcopy(client) for client in clients]

        print("Start server update...")
        # server side (all are based on syn data)
        for _ in range(args.interact_epoch):
            
            
            train_loaders = [DataLoader(trainset, batch_size=args.batch_size, shuffle=True) for trainset in CLIP_adapter.trainsets]
            trainloader_iters = [iter(dataloader) for dataloader in train_loaders]

            for batch_group in zip(*trainloader_iters):

                # mutual KD learning starts here
                #####################################################################################################
                
                # S.1 mutual KD loss
                for c in range(len(clients)):
                    clients[c].image_classifier.eval() 

                all_parameters = list(chain(*[adapter.parameters() for adapter in CLIP_adapter.adapters]))
                all_parameters += list(CLIP_adapter.MLP.parameters())
                optimizer = torch.optim.Adam(all_parameters, lr=args.lr)

                CLIP_adapter.MLP.train()

                loss_KD = 0
                for i, (images, labels, qualitys) in enumerate(batch_group):
                    images = images.to(device)
                    labels = labels.to(device)
                    qualitys = qualitys.to(device)

                    # 2.1 adapter is student
                    
                    # select all clients has the same category
                    clients_with_domain_i = category_to_clients[i]
                    image_classifiers_with_domain_i = [cp.deepcopy(clients[c].image_classifier) for c in clients_with_domain_i]
                    
                    # teacher_logits = avg(all clients with domain i logits)
                    teacher_logits = torch.zeros((images.shape[0], len(CLIP_adapter.class_names[i]))).to('cuda')
                    for img_classifer in image_classifiers_with_domain_i:
                        img_classifer.eval()
                        with torch.no_grad():
                            outputs = img_classifer(images)
                            teacher_logits += outputs

                    teacher_logits = teacher_logits/len(clients_with_domain_i)
                    teacher_logits *= qualitys.unsqueeze(1)

                    student_logits = CLIP_adapter.CLIP_logtis(i, True, images)

                    loss_KD += knowledge_distillation_loss(student_logits, teacher_logits, labels, alpha=.5)

                ########################################################################################################
                # insert interative learning for server
                ########################################################################################################
                loss_IL = 0
                for i, (images, labels, _) in enumerate(batch_group):
                    images = images.to(args.device)
                    labels = labels.to(args.device)

                    # S.2.1 calculate attension score
                    att_score = []
                    server_logits_all = []
                    for j in range(len(categories)):
                        server_logits_j = CLIP_adapter.CLIP_logtis(j, True, images)  
                        # print(server_logits_j.shape)
                        server_logits_all.append(server_logits_j)
                        att_score.append(CLIP_adapter.MLP(server_logits_j))

                        # CE loss -- correct adapter
                        if i == j: 
                            loss_IL += CE_loss(server_logits_j, labels)
                    
                    server_logits_all = torch.stack(server_logits_all, dim=0)
                    att_score = torch.stack(att_score, dim=0)
                    att_score = torch.softmax(att_score, dim=0)

                    # S.2.2 weighted sum of server logits
                    combined_logits = torch.sum(server_logits_all * att_score, dim=0)

                    # CE loss -- combined logits
                    loss_IL += CE_loss(combined_logits, labels)

                    # S.2.3 regularizaton
                    loss_IL += args.lam * (att_score - att_score[i]).max(dim=0)[0].sum()
                
                print(loss_KD.item(), loss_IL.item())
                loss = loss_KD + loss_IL
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                CLIP_adapter.MLP.eval()
                
                #  mutual KD for clients
                for c in range(len(clients)):
                    clients[c].image_classifier.train()
                optimizer = torch.optim.Adam(list(chain(*[client.image_classifier.parameters() for client in clients])), lr=args.lr)

                loss_KD = 0
                for i, (images, labels, qualitys) in enumerate(batch_group):
                    images = images.to(device)
                    labels = labels.to(device)
                    qualitys = qualitys.to(device)
                        
                    # 2.2 adapter is teacher

                    student_logits = clients[i].image_classifier(images)
                    student_logits *= qualitys.unsqueeze(1)

                    with torch.no_grad():
                        teacher_logits = CLIP_adapter.CLIP_logtis(i, False, images)

                    loss_KD += knowledge_distillation_loss(student_logits, teacher_logits, labels, alpha=.5)
                    
                optimizer.zero_grad()
                loss_KD.backward()
                optimizer.step()

                # mutual KD learning ends here
                #####################################################################################################

        # momentum update on clients
        for client, client_old in zip(clients, clients_copy):
            for param, param_old in zip(client.image_classifier.parameters(), client_old.image_classifier.parameters()):
                param.data = args.momentum * param_old.data + (1 - args.momentum) * param.data

        # testing
        # in-domain
        print('in-domain one2one testing...')
        CLIP_adapter.inference_one2one(epoch, categories)

        print('in-domain one2all testing...')
        CLIP_adapter.inference_one2all(epoch, categories, test_categories, in_domain=True)

        # out-domain
        print('out-domain one2all testing...')
        CLIP_adapter.inference_one2all(epoch, categories, test_categories, in_domain=False)

    
    print("--------Finish Communication round {}--------".format(epoch))



if __name__ == "__main__":
    main()
