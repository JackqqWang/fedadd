from torch.utils.data import random_split, DataLoader
import timm
import torch
from tqdm import tqdm

class Client(object):
    def __init__(self, dataset, num_classes, args):
        self.args = args
        self.device = args.device
        self.train_set, self.test_set = self.split_data(dataset)
        self.image_classifier = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
        self.image_classifier.to(self.device)


    def split_data(self, dataset):
        test_size = int(self.args.test_ratio * len(dataset))
        train_size = len(dataset) - test_size
        train, test = random_split(dataset, [train_size, test_size])

        return train, test
    

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


    def train_classifier(self):

        # local training
        optimizer = torch.optim.Adam(self.image_classifier.parameters(), lr=self.args.lr_local)
        train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_set, batch_size=self.args.batch_size, shuffle=False)

        for e in range(self.args.epoch):
            print(f"epoch {e}:")
            self.train(optimizer, train_loader)
            accuracy = self.test(test_loader)
            print(f"Accuracy: {accuracy:.2f}")