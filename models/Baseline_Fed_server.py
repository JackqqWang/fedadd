import torch.nn as nn
import clip
import torch
import os
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm import tqdm
from itertools import chain
from utils_FedAdapter import get_synthetic_dataset_server
from torch.utils.data import Dataset, DataLoader
import timm

## for baselines


CLIP_MODELS = {
    'RN50':'RN50',
    'RN101':'RN101',
    'RN50x4':'RN50x4',
    'RN50x16':'RN50x16',
    'RN50x64':'RN50x64',
    'ViT_B_32':'ViT-B/32',
    'ViT_B_16':'ViT-B/16',
    'ViT_L_14':'ViT-L/14',
    'ViT_L_14_336px':'ViT-L/14@336px'
}

# class CustomCLIPModel(nn.Module):
#     def __init__(self, clip_model, new_backbone):
#         super().__init__()
#         self.text_model = clip_model.text_model
#         self.visual_model = new_backbone  # Replace the visual model
#         self.logit_scale = clip_model.logit_scale
#         self.proj = clip_model.visual_projection  # You might need to adjust this too

#     def forward(self, image, text):
#         image_features = self.visual_model(image)
#         text_features = self.text_model(text)
#         # Rest of the forward logic adjusting dimensions as necessary
#         return image_features, text_features



class Server_CLIP(object):

    def __init__(self, args):
        self.clip_model, self.clip_preprocess = clip.load(
            CLIP_MODELS[args.model], device=args.device, jit=False)
        self.clip_model.eval()
        self.clip_model.encode_image = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=512)
        
        self.args = args
        self.device = args.device

        self.trainsets = []
        self.testsets = []
        self.class_names = []

        self.testsets_out = []
        self.class_names_out = []

        self.embed_dim = None
        self.output_dim = None

    def init_adapter(self):

        self.adapter = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.Tanh(
            ), nn.Linear(self.embed_dim, self.embed_dim), nn.Softmax(dim=1)).to(self.device)    
        

    def insert_datasets(self, real_dataset, classes, category):

        # real_dataset for testing on server
        self.testsets.append(real_dataset)

        # synthetic dataset for training adapters on server
        assert self.args.server_syn_size > 0, "server syn data cannot be 0"         
        train = get_synthetic_dataset_server(self.args, self.clip_preprocess, category)
        self.trainsets.append(train)
        self.class_names.append(classes)

        if self.embed_dim is None:
            self.output_dim = len(classes)

            data_loader = DataLoader(self.testsets[0], batch_size=self.args.batch_size, shuffle=False)
            for images, _ in data_loader:
                images = images.to(self.device)
                images_features = self.clip_model.encode_image(images)
                self.embed_dim = images_features.shape[1]
                break

    def insert_out_testsets(self, real_dataset, classes):
        # real_dataset for testing on server
        self.testsets_out.append(real_dataset)
        self.class_names_out.append(classes)

    def train(self, adapter, train_loader, optimizer, classes):

        adapter.train()

        loss_img = torch.nn.CrossEntropyLoss()
        loss_txt = torch.nn.CrossEntropyLoss()

        for images, labels in tqdm(train_loader):
            # Move the images and labels to the same device as the model
            images = images.to(self.device)
            labels = labels.to(self.device)

            # prompts
            text_descriptions = [f"This is a photo of a {classes[label]}" for label in labels]
            text_tokens = clip.tokenize(text_descriptions).to(self.device)

            # Encode images 
            image_features = self.clip_model.encode_image(images)
            image_features = image_features.to(dtype=torch.float32)
            image_features_att = adapter(image_features)
            image_features = torch.mul(image_features_att, image_features)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)

            # Encode text 
            text_features = self.clip_model.encode_text(text_tokens)
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
                len(images), dtype=torch.long, device=self.device)

            loss = (loss_img(logits_per_image, ground_truth) +
                    loss_txt(logits_per_text, ground_truth))/2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def test(self, adapter, test_loader, classes):
        adapter.eval()
        total_correct = 0
        total_images = 0

        with torch.no_grad(): 
            for images, labels in tqdm(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # prompts
                text_descriptions = [f"This is a photo of a {class_name}" for class_name in classes]
                text_tokens = clip.tokenize(text_descriptions).to(self.device)

                # # get logits
                # logits_per_image, logits_per_text = CLIP_apt.clip_model(images, text_tokens)
                # predictions = logits_per_image.softmax(dim=-1).argmax(dim=1)
                
                # Encode images 
                image_features = self.clip_model.encode_image(images)
                image_features = image_features.to(dtype=torch.float32)
                image_features_att = adapter(image_features)
                image_features = torch.mul(image_features_att, image_features)
                image_features /= image_features.norm(dim=-1, keepdim=True)            

                # Encode text 
                text_features = self.clip_model.encode_text(text_tokens)
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


    def train_adapter(self):

        trainsets = ConcatDataset([trainset for trainset in self.trainsets])
        train_loader = DataLoader(trainsets, batch_size=self.args.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=self.args.lr_CLIP)

        for epoch in range(self.args.interact_epoch):
            print(f"Epoch {epoch}:")
            self.train(self.adapter, train_loader, optimizer, self.class_names[0])


class Client_CLIP(object):

    def __init__(self, dataset, classes, args):

        self.args = args
        self.device = args.device
        
        self.clip_model, self.clip_preprocess = clip.load(
            CLIP_MODELS[args.model], device=args.device, jit=False)
        self.clip_model.eval()
        self.clip_model.encode_image = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=512)

        self.insert_dataset(dataset, classes)
        self.init_adapter()


    def split_data(self, dataset):
        test_size = int(self.args.test_ratio * len(dataset))
        train_size = len(dataset) - test_size
        train, test = random_split(dataset, [train_size, test_size])

        return train, test
    
    def insert_dataset(self, dataset, classes):

        self.trainset, self.testset = self.split_data(dataset)
        self.class_names = classes

        self.output_dim = len(classes)

        data_loader = DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=False)
        for images, _ in data_loader:
            images = images.to(self.device)
            images_features = self.clip_model.encode_image(images)
            self.embed_dim = images_features.shape[1]
            break

    def init_adapter(self):

        self.adapter = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.Tanh(
            ), nn.Linear(self.embed_dim, self.embed_dim), nn.Softmax(dim=1)).to(self.device)    
        

    def train(self, adapter, train_loader, optimizer, classes):

        adapter.train()

        loss_img = torch.nn.CrossEntropyLoss()
        loss_txt = torch.nn.CrossEntropyLoss()

        for images, labels in tqdm(train_loader):
            # Move the images and labels to the same device as the model
            images = images.to(self.device)
            labels = labels.to(self.device)

            # prompts
            text_descriptions = [f"This is a photo of a {classes[label]}" for label in labels]
            text_tokens = clip.tokenize(text_descriptions).to(self.device)

            # Encode images 
            image_features = self.clip_model.encode_image(images)
            image_features = image_features.to(dtype=torch.float32)
            image_features_att = adapter(image_features)
            image_features = torch.mul(image_features_att, image_features)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)

            # Encode text 
            text_features = self.clip_model.encode_text(text_tokens)
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
                len(images), dtype=torch.long, device=self.device)

            loss = (loss_img(logits_per_image, ground_truth) +
                    loss_txt(logits_per_text, ground_truth))/2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def test(self, adapter, test_loader, classes):
        adapter.eval()
        total_correct = 0
        total_images = 0

        with torch.no_grad(): 
            for images, labels in tqdm(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # prompts
                text_descriptions = [f"This is a photo of a {class_name}" for class_name in classes]
                text_tokens = clip.tokenize(text_descriptions).to(self.device)

                # # get logits
                # logits_per_image, logits_per_text = CLIP_apt.clip_model(images, text_tokens)
                # predictions = logits_per_image.softmax(dim=-1).argmax(dim=1)
                
                # Encode images 
                image_features = self.clip_model.encode_image(images)
                image_features = image_features.to(dtype=torch.float32)
                image_features_att = adapter(image_features)
                image_features = torch.mul(image_features_att, image_features)
                image_features /= image_features.norm(dim=-1, keepdim=True)            

                # Encode text 
                text_features = self.clip_model.encode_text(text_tokens)
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

    def train_adapter(self):

        train_loader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=False)
        
        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=self.args.lr_CLIP)

        for epoch in range(self.args.epoch):
            print(f"Epoch {epoch}:")
            self.train(self.adapter, train_loader, optimizer, self.class_names)
            self.test(self.adapter, test_loader, self.class_names)


class Server_ViT(object):

    def __init__(self, args, num_classes, clip_preprocess):

        self.args = args
        self.device = args.device

        self.clip_preprocess = clip_preprocess

        self.image_classifier = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
        self.image_classifier.to(self.device)
        
        self.trainsets = []
        self.testsets = []
        self.class_names = []

        self.testsets_out = []
        self.class_names_out = []
        

    def insert_datasets(self, real_dataset, classes, category):

        # real_dataset for testing on server
        self.testsets.append(real_dataset)

        # synthetic dataset for training adapters on server
        assert self.args.server_syn_size > 0, "server syn data cannot be 0"         
        train = get_synthetic_dataset_server(self.args, self.clip_preprocess, category)
        self.trainsets.append(train)
        self.class_names.append(classes)

    def insert_out_testsets(self, real_dataset, classes):
        # real_dataset for testing on server
        self.testsets_out.append(real_dataset)
        self.class_names_out.append(classes)

    def train(self, optimizer, train_loader):
        criterion = torch.nn.CrossEntropyLoss()

        self.image_classifier.train()

        for images, labels in tqdm(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.image_classifier(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def test(self, test_loader):

        self.image_classifier.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.image_classifier(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Test Accuracy: {correct / total}")

        return correct / total


    def train_ViT(self):

        trainsets = ConcatDataset([trainset for trainset in self.trainsets])
        train_loader = DataLoader(trainsets, batch_size=self.args.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.image_classifier.parameters(), lr=self.args.lr_CLIP)

        for epoch in range(self.args.interact_epoch):
            print(f"Epoch {epoch}:")
            self.train(optimizer, train_loader)


class Client_ViT(object):

    def __init__(self, dataset, classes, num_classes, args):

        self.args = args
        self.device = args.device
        
        self.image_classifier = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
        self.image_classifier.to(self.device)

        self.insert_dataset(dataset, classes)


    def split_data(self, dataset):
        test_size = int(self.args.test_ratio * len(dataset))
        train_size = len(dataset) - test_size
        train, test = random_split(dataset, [train_size, test_size])

        return train, test
    
    def insert_dataset(self, dataset, classes):

        self.trainset, self.testset = self.split_data(dataset)
        self.class_names = classes

    def train(self, optimizer, train_loader):
        criterion = torch.nn.CrossEntropyLoss()

        self.image_classifier.train()

        for images, labels in tqdm(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.image_classifier(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    
    def test(self, test_loader):

        self.image_classifier.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.image_classifier(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def train_ViT(self):

        train_loader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=False)
        
        optimizer = torch.optim.Adam(self.image_classifier.parameters(), lr=self.args.lr_CLIP)

        for epoch in range(self.args.epoch):
            print(f"Epoch {epoch}:")
            self.train(optimizer, train_loader)
            self.test(test_loader)