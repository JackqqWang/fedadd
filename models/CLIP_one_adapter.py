# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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


def get_image_features(image, model, cpreprocess, device='cuda', need_preprocess=False):
    if need_preprocess:
        image = cpreprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features

def freeze_param(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


CLIP_MODELS = {
    'RN50':'RN50',
    'RN101':'RN101',
    'RN50x4':'RN50x4',
    'RN50x16':'RN50x16',
    'RN50x64':'RN50x64',
    'ViT_S':'ViT-B/32',
    'ViT_B_32':'ViT-B/32',
    'ViT_B_16':'ViT-B/16',
    'ViT_L_14':'ViT-L/14',
    'ViT_L_14_336px':'ViT-L/14@336px'
}

class ClipModelatFed(object):

    def __init__(self, args, adp=True):
        self.clip_model, self.clip_preprocess = clip.load(
            CLIP_MODELS[args.model], device=args.device, jit=False)
        if args.model == 'ViT_S':
            self.clip_model.encode_image = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=512)

        self.clip_model.eval()
        self.args = args
        self.adp = adp
        self.device = args.device
        self.adapters = []
        self.trainsets = []
        self.testsets = []
        self.testsets_out = []
        self.class_names_out = []
        self.class_names = []
        self.best_acc = []
        self.embed_dim = None
        self.output_dim = None
    
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

        # print(f'total # of {category} server test data (real): {len(self.testsets[-1])}')
        # print(f'total # of {category} server train data (syn): {len(self.trainsets[-1])}')

    def insert_out_testsets(self, real_dataset, classes):
        # real_dataset for testing on server
        self.testsets_out.append(real_dataset)
        self.class_names_out.append(classes)

    def init_adapter(self):

        self.adapter = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim), 
            nn.Tanh(), 
            nn.Linear(self.embed_dim, self.embed_dim), 
            nn.Linear(self.embed_dim, self.embed_dim), 
            nn.Tanh(), 
            nn.Linear(self.embed_dim, self.embed_dim), 
            nn.Linear(self.embed_dim, self.embed_dim), 
            nn.Tanh(), 
            nn.Linear(self.embed_dim, self.embed_dim), 
            nn.Softmax(dim=1)
            ).to(self.device)    


    def test(self, adapter, test_loader, classes):
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


    def train_adapter(self, local_data, categories, test_categories):

        if 'real' in self.args.mode:
            # mix server syn data with local data
            trainsets = [ConcatDataset([local_data[i], self.trainsets[i]]) for i in range(len(self.trainsets))]
        else:
            trainsets = self.trainsets

        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=self.args.centralized_lr)
        loss_img = torch.nn.CrossEntropyLoss()
        loss_txt = torch.nn.CrossEntropyLoss()

        for epoch in range(self.args.centralized_epochs):
            print(f"Epoch {epoch}:")

            train_loaders = [DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True) for trainset in trainsets]
            trainloader_iters = [iter(dataloader) for dataloader in train_loaders]
            
            # training, all train sets -> adapter
            self.adapter.train()
            print('training')

            for batch_group in tqdm(zip(*trainloader_iters), total=len(train_loaders[0])):
            # for batch_group in zip(*trainloader_iters):

                loss = 0
                for i, (images, labels) in enumerate(batch_group):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # prompts
                    text_descriptions = [f"This is a photo of a {self.class_names[i][label]}" for label in labels]
                    text_tokens = clip.tokenize(text_descriptions).to(self.device)

                    # Encode images 
                    image_features = self.clip_model.encode_image(images)
                    image_features = image_features.to(dtype=torch.float32)
                    image_features_att = self.adapter(image_features)
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

                    loss += (loss_img(logits_per_image, ground_truth) +
                            loss_txt(logits_per_text, ground_truth))/2
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            self.adapter.eval()
            # in domain test
            for i in range(len(self.testsets)):
                print(f'in domain testing on {categories[i]}')
                test_loader = DataLoader(self.testsets[i], batch_size=self.args.batch_size, shuffle=False)
                self.test(self.adapter, test_loader, self.class_names[i])
            
            # out domain test
            for i in range(len(self.testsets_out)):
                print(f'out domain testing on {test_categories[i]}')
                test_loader = DataLoader(self.testsets_out[i], batch_size=self.args.batch_size, shuffle=False)
                self.test(self.adapter, test_loader, self.class_names_out[i])

