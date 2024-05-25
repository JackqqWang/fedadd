import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel, AdamW
from tqdm.auto import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for CLIP input size
    transforms.ToTensor(),  # Converts to [0, 1] and does not normalize
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

optimizer = AdamW(model.parameters(), lr=5e-6)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)

# Labels for CIFAR-10 dataset
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

epochs = 3
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        texts = [f"This is a photo of a {labels[target]}" for target in targets.tolist()]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(device)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # Image logits

        # Transform logits_per_image to match the target's shape
        # Assuming one correct label per image, we select the diagonal of logits_per_image
        # This works under the assumption that the corresponding texts for each image are the correct classes
        correct_logits = logits_per_image.diag()

        loss = torch.nn.functional.cross_entropy(correct_logits.unsqueeze(1), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=loss.item())

def evaluate(model, data_loader, processor, device):
    model.eval()
    total = 0
    correct = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Generate text inputs for CLIP using the class labels
            texts = [f"This is a photo of a {labels[target]}" for target in targets.tolist()]
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
            
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            predictions = logits_per_image.argmax(dim=1)
            
            total += targets.size(0)
            correct += (predictions == targets).sum().item()

    accuracy = 100 * correct / total
    return accuracy

accuracy = evaluate(model, test_loader, processor, device)
print(f"Test Accuracy: {accuracy:.4f}%")