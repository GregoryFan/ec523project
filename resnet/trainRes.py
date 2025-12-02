import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock, ResNet

#Access Database
data_root = "split_ttv_dataset_type_of_plants"

train_dir = f"{data_root}/Train_Set_Folder"
val_dir = f"{data_root}/Validation_Set_Folder"
test_dir = f"{data_root}/Test_Set_Folder"

#ResNet18 Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("Setting Datasets..")

#Datasets and loaders
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#Training and Testing/Validation
def train_model(net, criterion, optimizer, trainloader, valloader, device, width, epochs = 10):

    #Save validation/losses
    losses = []
    accs = []
    val_losses = []
    
    #Save best model
    best_model = None
    best_epoch = 0
    best_acc = -np.inf

    #Saving Checkpoints
    os.makedirs("checkpoints", exist_ok=True)

    #Training Loop
    print("Starting Training..\n")
    for epoch in range(epochs):
        running_loss = 0.0
        net.train()

        for i, data in enumerate(trainloader):
    
            #Get inputs, labels and send them to device
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            #Zero the gradient
            optimizer.zero_grad()

            #Forward Pass
            output = net(inputs)

            #Generate Loss and Backward
            loss = criterion(output, labels)
            loss.backward()

            #Step
            optimizer.step()

            #Update Loss
            running_loss += loss.item()

        #Print Loss per epoch
        avg_loss = running_loss / len(trainloader)
        print(f"Loss of Epoch {epoch+1}: {running_loss}, Average: {avg_loss:.2f} \n")

        #Run Validation
        net.eval()
        acc, val_loss = test_model(net, valloader, device, criterion)

        #Update and save best models 
        if acc > best_acc:
            best_model = copy.deepcopy(net)
            best_epoch = epoch
            best_acc = acc
            torch.save(best_model.state_dict(), f"checkpoints/best_resnet18w{width}.pth")
            print(f"New best model saved at epoch {epoch+1} with accuracy {acc:.2f}%\n")

        
        print(f"Accuracy of Epoch {epoch+1}: {acc}")

        #Save acc and loss
        losses.append(avg_loss)
        accs.append(acc)
        val_losses.append(val_loss)
    
    print("Finished Training")

    #Save last model
    torch.save(net.state_dict(), f"checkpoints/last_resnet18w{width}.pth")
    
    return net, best_model, best_epoch, best_acc, losses, accs, val_losses

def test_model(net, testloader, device, criterion = None):
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
      for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        output = net(inputs)

        if criterion != None:
            loss = criterion(output, labels)
            running_loss += loss.item()

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    avg_val_loss = running_loss / len(testloader) if criterion is not None else 0.0
    return acc, avg_val_loss


#Custom ResNet
def resnet18_scaled(width_mult=1.0, num_classes=30):
    base_channels = int(64 * width_mult)

    class ScaledResNet(ResNet):
        def __init__(self):
            super().__init__(block=BasicBlock, layers=[2, 2, 2, 2])
            self.inplanes = base_channels

            self.conv1 = nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(base_channels)

            self.layer1 = self._make_layer(BasicBlock, base_channels, 2)
            self.layer2 = self._make_layer(BasicBlock, base_channels * 2, 2, stride=2)
            self.layer3 = self._make_layer(BasicBlock, base_channels * 4, 2, stride=2)
            self.layer4 = self._make_layer(BasicBlock, base_channels * 8, 2, stride=2)

            final_channels = base_channels * 8 * BasicBlock.expansion
            self.fc = nn.Linear(final_channels, num_classes)

    return ScaledResNet()

#Training

widths = [0.5, 1, 2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = {}

for width in widths:

    print(f"Training with width {width}:\n")

    model = resnet18_scaled(width_mult = width, num_classes=30)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model, best_model, best_epoch, best_acc, losses, accs, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, device, width, 300)

    print(f"Best model with width {width} had accuracy {best_acc} at epoch {best_epoch}\n")

    print("Testing last model:\n")
    acc, _ = test_model(model, test_loader, device)
    print(f"Accuracy: {acc}\n")

    print("Testing best model:\n")
    acc, _ = test_model(best_model, test_loader, device)
    print(f"Accuracy: {acc}\n")

    results[width] = {
        "losses": losses,
        "accs": accs,
        "val_losses": val_losses,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
    }


#Plotting
plt.figure(figsize=(12, 5))

# Training Loss
plt.subplot(1, 3, 1)
for width in widths:
    plt.plot(results[width]["losses"], label=f"Width={width}")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss")
plt.legend()

# Validation Loss
plt.subplot(1, 3, 2)
for width in widths:
    plt.plot(results[width]["val_losses"], label=f"Width={width}")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss")
plt.legend()

# Validation Accuracy
plt.subplot(1, 3, 3)
for width in widths:
    plt.plot(results[width]["accs"], label=f"Width={width}")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")

print("Failsafe for losses/accs:")
for width in widths:
    result = results[width]
    print(f"Width: {width}\n")
    print(f"Losses: {result['losses']}\n")
    print(f"Accuracies: {result['accs']}\n")
    print(f"Validation Loss: {result['val_losses']}")




