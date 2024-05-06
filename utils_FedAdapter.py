import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
import random
import numpy as np
import os
import clip
import argparse
from tqdm import tqdm
from collections import defaultdict
from torch.nn.functional import kl_div, softmax, log_softmax, cross_entropy
from options_fedadapter import *

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy
    torch.manual_seed(seed_value)  # PyTorch
    torch.cuda.manual_seed(seed_value)  # PyTorch, for CUDA
    torch.cuda.manual_seed_all(seed_value)  # PyTorch, if using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Python hash build-in



def get_office_home_dataset(args, preprocess):

    data_root = os.path.join(args.data_dir, args.dataset)
    data_root = os.path.join(data_root, 'real')

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
    data_root = os.path.join(args.data_dir, args.dataset)
    data_root = os.path.join(data_root, 'real')

    print(f'loading real data from {data_root}')

    server = []
    local = []
    classes = []

    for _, category in enumerate(categories):
        dataset = datasets.ImageFolder(root=data_root + '/' + category, transform=preprocess)
        class_names = dataset.classes

        # print(f'total # of {category} data: {len(dataset)}')

        # split the dataset for server and local
        server_size = int(args.server_ratio * len(dataset))
        local_size = len(dataset) - server_size
        server_dataset, local_dataset = random_split(dataset, [server_size, local_size])

        server.append(server_dataset)
        local.append(local_dataset)
        classes.append(class_names)

        # print(f'total # of {category} server data: {len(server_dataset)}')
        # print(f'total # of {category} local data: {len(local_dataset)}')
    
    return server, local, classes


def get_synthetic_dataset_server(args, preprocess, category):
    data_root = os.path.join(args.data_dir, args.dataset)
    data_root = os.path.join(data_root, f'fake_{args.server_syn_ver}')

    print(f'loading synthetic data from {data_root}')
    
    dataset = datasets.ImageFolder(root=data_root + '/' + category, transform=preprocess)

    remaining = len(dataset) - args.server_syn_size
    server_dataset, _ = random_split(dataset, [args.server_syn_size, remaining])

    return server_dataset


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


# for centralized training
def centralized_zero_shot(CLIP_adapter, i, args):

    test_loader = DataLoader(CLIP_adapter.testsets[i], batch_size=args.batch_size, shuffle=False)

    classes = CLIP_adapter.class_names[i]

    total_correct = 0
    total_images = 0

    with torch.no_grad(): 
        for images, labels in tqdm(test_loader):
            images = images.to(args.device)
            labels = labels.to(args.device)

            # prompts
            text_descriptions = [f"This is a photo of a {class_name}" for class_name in classes]
            text_tokens = clip.tokenize(text_descriptions).to(args.device)

            # # get logits
            # logits_per_image, logits_per_text = CLIP_apt.clip_model(images, text_tokens)
            # predictions = logits_per_image.softmax(dim=-1).argmax(dim=1)
            
            # Encode images 
            image_features = CLIP_adapter.clip_model.encode_image(images)
            image_features = image_features.to(dtype=torch.float32)
            image_features /= image_features.norm(dim=-1, keepdim=True)            

            # Encode text 
            text_features = CLIP_adapter.clip_model.encode_text(text_tokens)
            text_features = text_features.to(dtype=torch.float32)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate the similarity 
            similarity = (100.0 * image_features @ text_features.T)
            predictions = similarity.argmax(dim=1)  # logits per image

            # Update the total correct predictions and total images processed
            total_correct += (predictions == labels).sum().item()
            total_images += images.size(0)

    # Calculate the accuracy
    accuracy = total_correct / total_images
    print(f"Test Accuracy: {accuracy:.2f}")

    return accuracy


def centralized_train(CLIP_adapter, local_data, i, category, args):

    if 'real' in args.mode:
        # mix server syn data with local data
        train_set = ConcatDataset([local_data[i], CLIP_adapter.trainsets[i]])
    else:
        train_set = CLIP_adapter.trainsets[i]

    print(f'train size: {len(train_set)}')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(CLIP_adapter.testsets[i], batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(CLIP_adapter.adapters[i].parameters(), lr=args.centralized_lr)

    model_path = os.path.join(args.data_dir, 'centralized_training', args.mode, f'{args.dataset}_fake_{args.server_syn_ver}')

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_path = os.path.join(model_path, f'adapter_epochs[{args.centralized_epochs}]_category[{category}].pth')

    if not os.path.exists(model_path):

        print(f"train adapter in domain {i}")
        for epoch in range(args.centralized_epochs):
            print(f"Epoch {epoch}:")
            CLIP_adapter.train(CLIP_adapter.adapters[i], train_loader, optimizer, CLIP_adapter.class_names[i])
            CLIP_adapter.test(CLIP_adapter.adapters[i], test_loader, CLIP_adapter.class_names[i])

        print(f"Save adapter in domain {i} to {model_path}")
        torch.save(CLIP_adapter.adapters[i].state_dict(), model_path)
    
    else:
        print(f"Load adapter in domain {i} from {model_path}")
        CLIP_adapter.adapters[i].load_state_dict(torch.load(model_path))
        CLIP_adapter.test(CLIP_adapter.adapters[i], test_loader, CLIP_adapter.class_names[i])

    return


'''
def test(CLIP_apt, test_loader, classes):

    CLIP_apt.clip_model.eval()  # Set the model to evaluation mode
    if CLIP_apt.adp:
        CLIP_apt.img_adap.eval()
    total_correct = 0
    total_images = 0
    device = CLIP_apt.device

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
                image_features_att = CLIP_apt.img_adap(image_features)
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
    

def train(CLIP_apt, train_loader, classes, optimizer):

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


def CLIP_logtis(CLIP_apt, images, classes):

    # prompts
    text_descriptions = [f"This is a photo of a {class_name}" for class_name in classes]
    text_tokens = clip.tokenize(text_descriptions).to(CLIP_apt.device)

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

    device = CLIP_apt.device

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


def transformer_train(model, train_loader, optimizer, device='cuda'):

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


def transformer_test(model, test_loader, device='cuda'):

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
'''