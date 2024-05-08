# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
import clip
import torch
import os
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from itertools import chain
from utils_FedAdapter import get_synthetic_dataset_server


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
    'ViT_B_32':'ViT-B/32',
    'ViT_B_16':'ViT-B/16',
    'ViT_L_14':'ViT-L/14',
    'ViT_L_14_336px':'ViT-L/14@336px'
}

'''
class ClipModelat(object):

    def __init__(self, model_name, adp=True, device='cuda'):
        self.clip_model, self.clip_preprocess = clip.load(
            CLIP_MODELS[model_name], device=device, jit=False)
        self.adp = adp
        self.device = device

    def init_adapter(self, data_loader):
        for images, _ in data_loader:
            images = images.to(self.device)
            images_features = self.clip_model.encode_image(images)
            embed_dim = images_features.shape[1]
            break

        self.img_adap = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(
            ), nn.Linear(embed_dim, embed_dim), nn.Softmax(dim=1)).to(self.device)    
'''


class Adapter_Interaction(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Adapter_Interaction, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(1, input_dim, output_dim))  # define the trainable parameter
        self.relu = nn.ReLU()

    def forward(self, x):
        interaction = torch.mul(x, self.weights)
        interaction = torch.sum(interaction, dim=2)
        interaction = self.relu(interaction)

        return interaction


class ClipModelatFed(object):

    # server side in FedAdapter

    def __init__(self, args, adp=True):
        self.clip_model, self.clip_preprocess = clip.load(
            CLIP_MODELS[args.model], device=args.device, jit=False)
        self.clip_model.eval()
        self.args = args
        self.adp = adp
        self.device = args.device
        self.adapters = []
        self.trainsets = []
        self.testsets = []
        self.class_names = []
        self.best_acc = []
        self.embed_dim = None
        self.output_dim = None

    def insert_datasets_old(self, dataset, classes, category):

        test_size = int(self.args.test_ratio * len(dataset))
        train_size = len(dataset) - test_size
        train, test = random_split(dataset, [train_size, test_size])

        if self.args.server_syn_size > 0:
            train = get_synthetic_dataset_server(self.args, self.clip_preprocess, category)
        
        self.trainsets.append(train)
        self.testsets.append(test)
        # self.class_names.append(train.dataset.classes)
        self.class_names.append(classes)

        if self.embed_dim is None:
            self.output_dim = len(classes)

            data_loader = DataLoader(self.testsets[0], batch_size=self.args.batch_size, shuffle=False)
            for images, _ in data_loader:
                images = images.to(self.device)
                images_features = self.clip_model.encode_image(images)
                self.embed_dim = images_features.shape[1]
                break
    
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

    def init_adapter(self):
        
        # data_loader = DataLoader(self.testsets[category], batch_size=self.args.batch_size, shuffle=False)
        # for images, _ in data_loader:
        #     images = images.to(self.device)
        #     images_features = self.clip_model.encode_image(images)
        #     embed_dim = images_features.shape[1]
        #     break

        img_adap = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.Tanh(
            ), nn.Linear(self.embed_dim, self.embed_dim), nn.Softmax(dim=1)).to(self.device)    
        
        self.adapters.append(img_adap)
        self.best_acc.append([float('-inf'), -1])

    def init_adapters_weights(self):
        # self.adapters_wights  = nn.Sequential(nn.Linear(self.output_dim*len(self.class_names), self.output_dim), nn.ReLU())
        # self.adapters_wights  = nn.Sequential(nn.Linear(self.output_dim, len(self.class_names)), nn.ReLU())
        # print(self.adapters_wights)

        self.adapters_wights = Adapter_Interaction(self.output_dim, len(self.class_names))
        self.adapters_wights.to(self.device)

    def init_MLP(self):
        self.MLP = nn.Sequential(nn.Linear(self.output_dim, 1), nn.ReLU())
        self.MLP.to(self.device)
        

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


    def train_adapter(self, i, category, epochs, init=False):

        train_loader = DataLoader(self.trainsets[i], batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(self.testsets[i], batch_size=self.args.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(self.adapters[i].parameters(), lr=self.args.lr_CLIP)

        model_path = os.path.join(self.args.data_dir, 'init_adapter', self.args.dataset)
        if self.args.server_syn_size > 0:
            model_path = os.path.join(model_path, f'fake_{self.args.server_syn_ver}')
        else:
            model_path = os.path.join(model_path, 'real')

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if init:
            model_path = os.path.join(model_path, f'adapter_epochs[{epochs}]_category[{category}].pth')
        else:
            model_path = os.path.join(model_path, f'adapter_FT_epochs[{epochs}]_category[{category}].pth')

        if not os.path.exists(model_path):

            print(f"train adapter in domain {i}")
            for epoch in range(epochs):
                print(f"Epoch {epoch}:")
                self.train(self.adapters[i], train_loader, optimizer, self.class_names[i])
                self.test(self.adapters[i], test_loader, self.class_names[i])

            if init:
                print(f"Save adapter in domain {i} to {model_path}")
                torch.save(self.adapters[i].state_dict(), model_path)
            
        else:
            print(f"Load adapter in domain {i} from {model_path}")
            self.adapters[i].load_state_dict(torch.load(model_path))
            self.test(self.adapters[i], test_loader, self.class_names[i])
            

    def inference_all_adapters(self, e):
        for i in range(len(self.adapters)):
            test_loader = DataLoader(self.testsets[i], batch_size=self.args.batch_size, shuffle=False)
            test_acc_i = self.test(self.adapters[i], test_loader, self.class_names[i])
            print(f"After server update, we test adapter in domain {i} at round {e} has accuracy: {test_acc_i}")

            if test_acc_i > self.best_acc[i][0]:
                print("Start saving the best adapter for category...")
                self.best_acc[i] = [test_acc_i, e]
                torch.save(self.adapters[i].state_dict(), f'Adapters/adapter_domain[{i}]_round[{e}].pth')
                print("finish saving the best adapter for category.")
        
    
    def CLIP_logtis(self, i, student, images):

        if student:
            self.adapters[i].train()
        else:
            self.adapters[i].eval()

        # prompts
        text_descriptions = [f"This is a photo of a {class_name}" for class_name in self.class_names[i]]
        text_tokens = clip.tokenize(text_descriptions).to(self.device)

        # Encode images 
        image_features = self.clip_model.encode_image(images)
        image_features = image_features.to(dtype=torch.float32)
        image_features_att = self.adapters[i](image_features)
        # attention
        image_features = torch.mul(image_features_att, image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)            

        # Encode text 
        text_features = self.clip_model.encode_text(text_tokens)
        text_features = text_features.to(dtype=torch.float32)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate the similarity (logits)
        similarity = (100.0 * image_features @ text_features.T)

        return similarity


    def train_adapters_and_weights(self, switch):

        # switch is True, update adapters and fix weights
        # switch is False, update weights and fix adapters

        if switch:
            for adapter in self.adapters:
                adapter.train()
            self.adapters_wights.eval()
            all_parameters = list(chain(*[adapter.parameters() for adapter in self.adapters]))
            optimizer = torch.optim.Adam(all_parameters, lr=self.args.lr_CLIP)
        else:
            for adapter in self.adapters:
                adapter.eval()
            self.adapters_wights.train()
            optimizer = torch.optim.Adam(self.adapters_wights.parameters(), lr=self.args.lr_CLIP)

        criterion = torch.nn.CrossEntropyLoss()
            
        train_loaders = [DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True) for trainset in self.trainsets]
        trainloader_iters = [iter(dataloader) for dataloader in train_loaders]
  
        for batch_group in zip(*trainloader_iters):
            loss = 0

            for i, batch in enumerate(batch_group):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits_i = self.CLIP_logtis(i, switch, images)

                logits_all = []
                for j in range(len(self.adapters)):
                    logits_all.append(self.CLIP_logtis(j, switch, images))
                # logits_all = torch.cat(logits_all, dim=1)
                logits_all = torch.stack(logits_all, dim=2)
                logits_all = self.adapters_wights(logits_all)

                loss += criterion(logits_i, labels) + self.args.beta * criterion(logits_all, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def inference_weighted_adapters(self, e):
        self.adapters_wights.eval()
        for adapter in self.adapters:
            adapter.eval()

        total = [0]*len(self.class_names)
        correct = [0]*len(self.class_names)

        for i in range(len(self.class_names)):
            test_loader = DataLoader(self.testsets[i], batch_size=self.args.batch_size, shuffle=False)
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits_all = []
                for j in range(len(self.adapters)):
                    logits_all.append(self.CLIP_logtis(j, False, images))
                # logits_all = torch.cat(logits_all, dim=1)
                logits_all = torch.stack(logits_all, dim=2)
                logits_all = self.adapters_wights(logits_all)

                total[i] += labels.size(0)
                correct[i] += (logits_all.argmax(dim=1) == labels).sum().item()

        acc = [correct[i]/total[i] for i in range(len(self.class_names))]

        print(f"weighted adpater fusion accuracy at round {e}:")
        for i in range(len(acc)):
            print(f"domain {i} accuracy: {acc[i]}")
        print(f"Total accuracy: {sum(correct)/sum(total)}")

'''
class ClipModelat_old(object):

    CLIP_MODELS = [
        'RN50',
        'RN101',
        'RN50x4',
        'RN50x16',
        'RN50x64',
        'ViT-B/32',
        'ViT-B/16',
        'ViT-L/14',
        'ViT-L/14@336px'
    ]

    def __init__(self, model_name='Vit-B/32', device='cuda', logger=None, imgadpy=True, freezepy=True):
        self.logger = logger
        if type(model_name) is int:
            model_name = self.index_to_model(model_name)
        self.model, self.preprocess = clip.load(
            model_name, device=device, jit=False)
        self.model.eval()
        self.model.to(device)
        self.model_name = model_name
        self.imgadpy = imgadpy
        self.freezepy = freezepy
        self.device = device

    def initdgatal(self, dataloader):

        for batch in dataloader:
            image, _, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
            image_features = get_image_features(
                image, self.model, self.preprocess)
            break
        if self.freezepy:
            freeze_param(self.model)

        if self.imgadpy:
            self.img_adap = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]), nn.Tanh(
            ), nn.Linear(image_features.shape[1], image_features.shape[1]), nn.Softmax(dim=1)).to(self.device)

    def index_to_model(self, index):
        return self.CLIP_MODELS[index]

    @staticmethod
    def get_model_name_by_index(index):
        name = ClipModelat.CLIP_MODELS[index]
        name = name.replace('/', '_')
        return name

    def setselflabel(self, labels):
        self.labels = labels

'''