import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
import random
import numpy as np
import os
from models.CLIP_adapter import ClipModelat, ClipModelatFed
import clip
import argparse
from tqdm import tqdm
# from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTConfig
from torch.nn.functional import kl_div, softmax, log_softmax, cross_entropy
from collections import defaultdict
import timm
from central_option import *

args = arg_parser()


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy
    torch.manual_seed(seed_value)  # PyTorch
    torch.cuda.manual_seed(seed_value)  # PyTorch, for CUDA
    torch.cuda.manual_seed_all(seed_value)  # PyTorch, if using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Python hash build-in





def get_office_home_dataset(args, preprocess):

    data_root = '../../../data/jiaqi/OfficeHomeDataset_10072016'

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),  # Resize for CLIP input size
    #     transforms.ToTensor(),  # Converts to [0, 1] and does not normalize
    # ])

    dataset = datasets.ImageFolder(root=data_root + '/' + args.category, transform=preprocess)
    class_names = dataset.classes

    # split the dataset for server and local
    server_size = int(args.server_ratio * len(dataset))
    local_size = len(dataset) - server_size
    server_dataset, local_dataset = random_split(dataset, [server_size, local_size])

    # Split the dataset into train and test
    DL = []
    for dataset in [server_dataset, local_dataset]:
        train_size = int((1 - args.test_ratio) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        DL.append((train_loader, test_loader))

    return DL, class_names


def Split_office_home_dataset(args, preprocess, categories):
    data_root = '../../../data/jiaqi/OfficeHomeDataset_10072016'

    server = []
    local = []
    classes = []

    for _, category in enumerate(categories):
        dataset = datasets.ImageFolder(root=data_root + '/' + category, transform=preprocess)
        class_names = dataset.classes

        # split the dataset for server and local
        server_size = int(args.server_ratio * len(dataset))
        local_size = len(dataset) - server_size
        server_dataset, local_dataset = random_split(dataset, [server_size, local_size])

        server.append(server_dataset)
        local.append(local_dataset)
        classes.append(class_names)
    
    return server, local, classes


def distribute_local_data(local_data, num_clients, num_categories, args):
    client_datasets = [[] for _ in range(num_clients)]
    category_to_clients = defaultdict(set)
    client_to_categories = {}

    for client in range(num_clients):
        category = client % num_categories
        categories_for_client = [category]

        # # randomly select m categories for each client
        # categories_for_client = np.random.choice(range(num_categories), args.num_client_cat, replace=False)
        client_to_categories[client] = categories_for_client

        # Record the client under each category they now have
        for category in categories_for_client:
            category_to_clients[category].add(client)

    # # Assign each client their own category and m-1 random others
    # for client in range(num_clients):
    #     categories_for_client = [client]  # Start with their own category
    #     other_categories = list(set(range(num_clients)) - {client})  # Possible other categories
    #     categories_for_client.extend(np.random.choice(other_categories, args.num_categories-1, replace=False))
        
    #     # Record the client under each category they now have
    #     for category in categories_for_client:
    #         category_to_clients[category].add(client)
    
    for category in category_to_clients:
        category_to_clients[category] = list(category_to_clients[category])

    # Split and distribute each category's dataset among its clients
    for category, clients in category_to_clients.items():
        # splits = np.array_split(local_data[category], len(clients))  # Split the category dataset
        total_size = len(local_data[category])
        base_size = total_size // len(clients)
        remainder = total_size % len(clients)
        lengths = [base_size + 1 if i < remainder else base_size for i in range(len(clients))]
        splits = random_split(local_data[category], lengths)

        for client, split_data in zip(clients, splits):
            client_datasets[client].append(split_data)
    
    # Merge all datasets for each client into one
    for client in range(num_clients):
        client_datasets[client] = ConcatDataset(client_datasets[client])

    return client_datasets, dict(category_to_clients), client_to_categories



# def test_deleted(CLIP_apt, test_loader, classes):

#     CLIP_apt.clip_model.eval()  # Set the model to evaluation mode
#     if CLIP_apt.adp:
#         CLIP_apt.img_adap.eval()
#     total_correct = 0
#     total_images = 0
#     device = CLIP_apt.device

#     with torch.no_grad(): 
#         for images, labels in tqdm(test_loader):
#             images = images.to(device)
#             labels = labels.to(device)

#             # prompts
#             text_descriptions = [f"This is a photo of a {class_name}" for class_name in classes]
#             text_tokens = clip.tokenize(text_descriptions).to(device)

#             # # get logits
#             # logits_per_image, logits_per_text = CLIP_apt.clip_model(images, text_tokens)
#             # predictions = logits_per_image.softmax(dim=-1).argmax(dim=1)
            
#             # Encode images 
#             image_features = CLIP_apt.clip_model.encode_image(images)
#             image_features = image_features.to(dtype=torch.float32)
#             if CLIP_apt.adp:
#                 image_features_att = CLIP_apt.img_adap(image_features)
#                 image_features = torch.mul(image_features_att, image_features)
#             image_features /= image_features.norm(dim=-1, keepdim=True)            

#             # Encode text 
#             text_features = CLIP_apt.clip_model.encode_text(text_tokens)
#             text_features = text_features.to(dtype=torch.float32)
#             text_features /= text_features.norm(dim=-1, keepdim=True)

#             # Calculate the similarity 
#             similarity = (100.0 * image_features @ text_features.T)
#             predictions = similarity.argmax(dim=1)

#             # Update the total correct predictions and total images processed
#             total_correct += (predictions == labels).sum().item()
#             total_images += images.size(0)

#     # Calculate the accuracy
#     accuracy = total_correct / total_images
#     print(f"Accuracy: {accuracy:.2f}")

#     return accuracy
    

def test(CLIP_apt, i, test_loader, classes):

    CLIP_apt.clip_model.eval()  # Set the model to evaluation mode
    if CLIP_apt.adp:
        for adapter in CLIP_apt.adapters:
            adapter.eval()
    total_correct = 0
    total_images = 0
    device = args.device

    with torch.no_grad(): 
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # prompts
            text_descriptions = [f"This is a photo of a {class_name}" for class_name in classes]
            text_tokens = clip.tokenize(text_descriptions).to(device)

            # # get logits
            # logits_per_image, logits_per_text = CLIP_apt.clip_model(images, text_tokens)
            # predictions = logits_per_image.softmax(dim=-1).argmax(dim=1)
            
            # Encode images 
            image_features = CLIP_apt.clip_model.encode_image(images)
            image_features = image_features.to(dtype=torch.float32)
            if CLIP_apt.adp:
                image_features_att = CLIP_apt.adapters[i](image_features)
                image_features = torch.mul(image_features_att, image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)            

            # Encode text 
            text_features = CLIP_apt.clip_model.encode_text(text_tokens)
            text_features = text_features.to(dtype=torch.float32)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate the similarity 
            similarity = (100.0 * image_features @ text_features.T)
            predictions = similarity.argmax(dim=1)

            # Update the total correct predictions and total images processed
            total_correct += (predictions == labels).sum().item()
            total_images += images.size(0)

    # Calculate the accuracy
    accuracy = total_correct / total_images
    print(f"Accuracy: {accuracy:.2f}")

    return accuracy
   
def train(CLIP_apt, i, train_loader, classes, optimizer):

    if CLIP_apt.adp:
        CLIP_apt.clip_model.eval()
        CLIP_apt.adapters[i].train()
    else:
        CLIP_apt.clip_model.train()
    device = args.device

    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()

    for images, labels in tqdm(train_loader):
        # Move the images and labels to the same device as the model
        images = images.to(device)
        labels = labels.to(device)

        # prompts
        text_descriptions = [f"This is a photo of a {classes[label]}" for label in labels]
        text_tokens = clip.tokenize(text_descriptions).to(device)

        # Encode images 
        image_features = CLIP_apt.clip_model.encode_image(images)
        image_features = image_features.to(dtype=torch.float32)
        if CLIP_apt.adp:
            image_features_att = CLIP_apt.adapters[i](image_features)
            image_features = torch.mul(image_features_att, image_features)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)

        # Encode text 
        text_features = CLIP_apt.clip_model.encode_text(text_tokens)
        text_features = text_features.to(dtype=torch.float32)
        text_features = text_features/text_features.norm(dim=-1, keepdim=True)

        # Compute similartiy
        # logit_scale = CLIP_apt.clip_model.logit_scale.exp()
        # print(logit_scale)
        logit_scale = 100
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # Calculate the loss
        ground_truth = torch.arange(
            len(images), dtype=torch.long, device=device)

        loss = (loss_img(logits_per_image, ground_truth) +
                loss_txt(logits_per_text, ground_truth))/2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return



# def train_deleted(CLIP_apt, train_loader, classes, optimizer):

    if CLIP_apt.adp:
        CLIP_apt.img_adap.train()
        CLIP_apt.clip_model.eval()
    else:
        CLIP_apt.clip_model.train()
    device = CLIP_apt.device

    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()

    for images, labels in tqdm(train_loader):
        # Move the images and labels to the same device as the model
        images = images.to(device)
        labels = labels.to(device)

        # prompts
        text_descriptions = [f"This is a photo of a {classes[label]}" for label in labels]
        text_tokens = clip.tokenize(text_descriptions).to(device)

        # Encode images 
        image_features = CLIP_apt.clip_model.encode_image(images)
        image_features = image_features.to(dtype=torch.float32)
        if CLIP_apt.adp:
            image_features_att = CLIP_apt.img_adap(image_features)
            image_features = torch.mul(image_features_att, image_features)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)

        # Encode text 
        text_features = CLIP_apt.clip_model.encode_text(text_tokens)
        text_features = text_features.to(dtype=torch.float32)
        text_features = text_features/text_features.norm(dim=-1, keepdim=True)

        # Compute similartiy
        # logit_scale = CLIP_apt.clip_model.logit_scale.exp()
        # print(logit_scale)
        logit_scale = 100
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # Calculate the loss
        ground_truth = torch.arange(
            len(images), dtype=torch.long, device=device)

        loss = (loss_img(logits_per_image, ground_truth) +
                loss_txt(logits_per_text, ground_truth))/2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return


def knowledge_distillation_loss(output_student, output_teacher, labels, temperature=1, alpha=0.5):

    # Soften the outputs
    soft_logits_student = log_softmax(output_student / temperature, dim=1)
    soft_logits_teacher = softmax(output_teacher / temperature, dim=1)

    # Compute the soft target loss (KL Divergence)
    soft_target_loss = kl_div(soft_logits_student, soft_logits_teacher, reduction='batchmean') * (temperature ** 2)

    # Compute the hard target loss (Cross Entropy)
    hard_target_loss = cross_entropy(output_student, labels)

    loss = alpha * soft_target_loss + (1 - alpha) * hard_target_loss

    # Combine the losses
    return loss


def CLIP_logtis(CLIP_apt, images, classes):

    # prompts
    text_descriptions = [f"This is a photo of a {class_name}" for class_name in classes]
    text_tokens = clip.tokenize(text_descriptions).to(args.device)

    # Encode images 
    image_features = CLIP_apt.clip_model.encode_image(images)
    image_features = image_features.to(dtype=torch.float32)
    if CLIP_apt.adp:
        image_features_att = CLIP_apt.img_adap(image_features)
        # attention
        image_features = torch.mul(image_features_att, image_features)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)            

    # Encode text 
    text_features = CLIP_apt.clip_model.encode_text(text_tokens)
    text_features = text_features.to(dtype=torch.float32)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate the similarity (logits)
    similarity = (100.0 * image_features @ text_features.T)

    return similarity


def mul_distillation(args, CLIP_apt, image_classifier, train_loader, test_loader, classes):

    device = args.device

    for e in range(args.KD_epoch):

        # CLIP is student
        CLIP_apt.img_adap.train()
        CLIP_apt.clip_model.eval()
        image_classifier.eval()

        # optimizer
        optimizer = torch.optim.Adam(CLIP_apt.img_adap.parameters(), lr=args.lr_KD)

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # logits of CLIP (student logits)
            student_logits = CLIP_logtis(CLIP_apt, images, classes)

            # logits of transformer (teacher logits)
            with torch.no_grad():
                teacher_logits = image_classifier(images)

            # KD loss
            loss = knowledge_distillation_loss(student_logits, teacher_logits, labels, args.temperature, args.alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if args.mul_KD:
            # local classifier is student
            CLIP_apt.img_adap.eval()
            CLIP_apt.clip_model.eval()
            image_classifier.train()

            # optimizer
            optimizer = torch.optim.Adam(image_classifier.parameters(), lr=args.lr_KD)

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                # logits of transformer (student logits)
                student_logits = image_classifier(images)

                # logits of CLIP (teacher logits)
                with torch.no_grad():
                    teacher_logits = CLIP_logtis(CLIP_apt, images, classes)

                # KD loss
                loss = knowledge_distillation_loss(student_logits, teacher_logits, labels, args.temperature, args.alpha)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # test
        print(f'KD Epoch {e}:')
        test(CLIP_apt, test_loader, classes)
        
    return


def CLIP_zero_shot(args):

    # CLIP_apt = ClipModelat(model_name=args.model, device='cuda', adp=False)
    # # _, test_loader, class_names = get_office_home_dataset(args.category, CLIP_apt.clip_preprocess, batch_size=args.batch_size, split_ratio=0.8)
    # (_, (_, test_loader)), class_names = get_office_home_dataset(args, CLIP_apt.clip_preprocess)
    
    # test(CLIP_apt, test_loader, class_names)

    categories = ['Art', 'Clipart', 'Product', 'Real_World']

    # init CLIP 
    CLIP_adapter = ClipModelatFed(args, adp=True, device=args.device)

    # load and split data
    print('split data for server and clients')
    server_data, _, classes = Split_office_home_dataset(args, CLIP_adapter.clip_preprocess, categories)

    # insert data to server and init adapters
    print('create adpaters for each category and insert server data')
    for i in range(len(categories)):
        CLIP_adapter.insert_datasets(server_data[i], classes[i])
        CLIP_adapter.init_adapter(i)

    CLIP_adapter.adp = False    

    for i in range(len(categories)):
        print(f'zero-shot for category {categories[i]}...')
        print(len(CLIP_adapter.testsets[i]))
        test_loader = DataLoader(CLIP_adapter.testsets[i], batch_size=args.batch_size, shuffle=False)
        test(CLIP_adapter, i, test_loader, classes[i])
    
    return


# def CLIP_fine_tune_deleted(category, model_type='ViT-B/32'):

#     CLIP_apt = ClipModelat(model_name=model_type, device=args.device, adp=False)
#     train_loader, test_loader, class_names = get_office_home_dataset(category, CLIP_apt.clip_preprocess, batch_size=32, split_ratio=0.8)
#     optimizer = torch.optim.Adam(CLIP_apt.clip_model.parameters(), lr=1e-4)
#     for e in range(5):
#         train(CLIP_apt, train_loader, class_names, optimizer)
#         test(CLIP_apt, test_loader, class_names)


#     return


def CLIP_adapter_central(args):

    categories = ['Art', 'Clipart', 'Product', 'Real_World']

    # init CLIP 
    CLIP_adapter = ClipModelatFed(args, adp=args.adp, device=args.device) # adp = False, fine tune all the clip without adapters

    # load and split data
    print('split data for server and clients')
    server_data, local_data, classes = Split_office_home_dataset(args, CLIP_adapter.clip_preprocess, categories)

    # insert data to server and init adapters
    print('create adpaters for each category and insert server data')
    for i in range(len(categories)):
        CLIP_adapter.insert_datasets(server_data[i], classes[i])
        if CLIP_adapter.adp:
            CLIP_adapter.init_adapter(i)

    # initializataion
    if CLIP_adapter.adp:
        for i in range(len(categories)):
            print(f"Initialize adapter for category {categories[i]}")
            CLIP_adapter.train_adapter(i, args.adp_init_epoch)

    # fine-tune
    client_datasets, _, _ = distribute_local_data(local_data, len(categories), len(categories), args)
    for i in range(len(categories)):
        print(f"Fine-tune adapter for category {categories[i]}")
        for epoch in range(args.communication_rounds):
            print(f"fine-tuning round {epoch}")
            test_size = int(args.test_ratio * len(client_datasets[i]))
            train_size = len(client_datasets[i]) - test_size
            train_set, test_set = random_split(client_datasets[i], [train_size, test_size])
            
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

            if CLIP_adapter.adp:
                optimizer = torch.optim.Adam(CLIP_adapter.adapters[i].parameters(), lr=args.lr_local)
            else:
                optimizer = torch.optim.Adam(CLIP_adapter.clip_model.parameters(), lr=args.lr_local)

            train(CLIP_adapter, i, train_loader, classes[i], optimizer)
            test(CLIP_adapter, i, test_loader, classes[i])

    return



def transformer_train(model, train_loader, optimizer, device=args.device):

    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def transformer_test(model, test_loader, device=args.device):

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


# def CLIP_adapter_deleted(args):

    print('loading data, CLIP, and transformer...')

    # init CLIP and load data
    CLIP_apt = ClipModelat(model_name=args.model, device=args.device, adp=True)
    [(server_train_loader, server_test_loader), (local_train_loader, local_test_loader)], class_names = get_office_home_dataset(args, CLIP_apt.clip_preprocess)
    
    # print(len(server_train_loader.dataset), len(server_test_loader.dataset), len(local_train_loader.dataset), len(local_test_loader.dataset))
    # exit()

    CLIP_apt.init_adapter(server_test_loader)
    optimizer = torch.optim.Adam(CLIP_apt.img_adap.parameters(), lr=args.lr_CLIP)

    # init transformer
    # config = ViTConfig.from_pretrained('google/vit-tiny-patch16-224-in21k')
    # config.num_labels = len(class_names)
    # image_classifier = ViTForImageClassification(config)
    
    image_classifier = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=len(class_names))
    image_classifier.to(device=args.device)
    optimizer_classifier = torch.optim.Adam(image_classifier.parameters(), lr=args.lr_local)

    # train transformer
    print('training transformer...')
    best_acc = float('-inf')
    for _ in range(args.epoch):
        transformer_train(image_classifier, local_train_loader, optimizer_classifier)
        
        test_acc = transformer_test(image_classifier, local_test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(image_classifier.state_dict(), f'Transformers/Image_classifier_{args.category}.pth')
            print(f"Best accuracy: {best_acc:.2f}")

    # train CLIP adapter
    print('training CLIP adapter...')
    best_acc = float('-inf')
    for _ in range(args.epoch):
        train(CLIP_apt, server_train_loader, class_names, optimizer)
        
        mul_distillation(args, CLIP_apt, image_classifier, server_train_loader, server_test_loader, class_names)
        
        test_acc = test(CLIP_apt, server_test_loader, class_names)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(CLIP_apt.img_adap.state_dict(), f'Adapters/CLIP_Adapter_{args.category}_{args.model}.pth')
            print(f"Best accuracy: {best_acc:.2f}")

    return



if __name__ == "__main__":
    args = arg_parser()
    set_seed(args.seed)
    CLIP_zero_shot(args)
    # if args.zero_shot:
    # CLIP_adapter_central(args)
    # CLIP_fine_tune(category='Art', model_type='ViT-B/32')
    # CLIP_adapter(args)