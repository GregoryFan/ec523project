import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
from torchvision.models import resnet18

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
def train_model(net, criterion, optimizer, trainloader, valloader, device, epochs = 10):

    #Save validation/losses
    losses = []
    accs = []
    
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
        acc = test_model(net, val_loader, device)

        #Update and save best models 
        if acc > best_acc:
            best_model = copy.deepcopy(net)
            best_epoch = epoch
            best_acc = acc
            torch.save(best_model.state_dict(), "checkpoints/best_resnet18.pth")
            print(f"New best model saved at epoch {epoch+1} with accuracy {acc:.2f}%\n")

        
        print(f"Accuracy of Epoch {epoch+1}: {acc}")

        #Save acc and loss
        losses.append(avg_loss)
        accs.append(acc)
    
    print("Finished Training")

    #Save last model
    torch.save(net.state_dict(), "checkpoints/last_resnet18.pth")
    
    return net, best_model, best_epoch, best_acc, losses, accs

def test_model(net, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
      for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        output = net(inputs)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc

#Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights = None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 30)
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

#Plotting
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
plt.savefig("training_curves.png")

print("Failsafe for losses/accs:")
print("Losses:", losses)
print("Accuracies:", accs)




