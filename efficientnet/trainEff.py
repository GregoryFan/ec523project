import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
from torchvision.models import efficientnet_b0
from PIL import Image, UnidentifiedImageError

def safe_loader(path):
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        print(f"Skipping invalid image: {path}")
        return None

#Access Database
data_root = "../../Composite data/Composite data"

train_dir = f"{data_root}/Training"
val_dir = f"{data_root}/Validation"
test_dir = f"{data_root}/Testing"

# EfficientNet Preprocessing 224 x 224 same as
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("Setting Datasets..")

class FilteredImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # Get the path and label before loading
        path, label = self.samples[index]

        # Try to load the image
        sample = self.loader(path)

        # If loading failed (returns None), skip to next valid image
        if sample is None:
            # Find next valid index (circular)
            next_index = (index + 1) % len(self.samples)
            return self.__getitem__(next_index)

        # Apply transform only if we have a valid image
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return sample, label

#Datasets and loaders
train_dataset = FilteredImageFolder(train_dir, transform=transform, loader=safe_loader)
val_dataset = FilteredImageFolder(val_dir, transform=transform, loader=safe_loader)
test_dataset = FilteredImageFolder(test_dir, transform=transform, loader=safe_loader)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training and Validation
def train_model(net, criterion, optimizer, trainloader, valloader, device, epochs=10):
    losses = []
    accs = []
    best_model = None
    best_epoch = 0
    best_acc = -np.inf

    os.makedirs("checkpoints", exist_ok=True)
    print("Starting Training...\n")

    for epoch in range(epochs):
        running_loss = 0.0
        net.train()

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Validation accuracy
        net.eval()
        acc = test_model(net, valloader, device)

        if acc > best_acc:
            best_model = copy.deepcopy(net)
            best_epoch = epoch
            best_acc = acc
            torch.save(best_model.state_dict(), "checkpoints/best_efficientnetb0.pth")
            print(f"New best model saved at epoch {epoch+1} with accuracy {acc:.2f}%\n")

        losses.append(avg_loss)
        accs.append(acc)

    torch.save(net.state_dict(), "checkpoints/last_efficientnetb0.pth")
    print("Training Finished!\n")
    return net, best_model, best_epoch, best_acc, losses, accs

def test_model(net, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model setup (EfficientNet-B0)
model = efficientnet_b0(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 30)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

model, best_model, best_epoch, best_acc, losses, accs = train_model(model, criterion, optimizer, train_loader, val_loader, device, 300)

print(f"Best model had accuracy {best_acc} at epoch {best_epoch}\n")

print("Testing last model:\n")
acc = test_model(model, test_loader, device)
print(f"Accuracy: {acc}\n")

print("Testing best model:\n")
acc = test_model(best_model, test_loader, device)
print(f"Accuracy: {acc}\n")

# Plot curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves_efficientnet.png")

print("Failsafe for losses/accs:")
print("Losses:", losses)
print("Accuracies:", accs)