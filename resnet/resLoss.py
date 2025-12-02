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
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

#Resnet Model
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

#Load resnet18 models
widths = [0.5, 1.0, 2.0]
model_files = {
    0.5: "last_resnet18w0.5.pth",
    1.0: "last_resnet18w1.pth",
    2.0: "last_resnet18w2.pth",
}

#Run only test on models
def test_model_loss(net, testloader, device, criterion = None):
    net.eval()
    running_loss = 0.0

    with torch.no_grad():
      for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        output = net(inputs)

        if criterion is not None:
            loss = criterion(output, labels)
            running_loss += loss.item()

    avg_val_loss = running_loss / len(testloader) if criterion is not None else 0.0
    return avg_val_loss

#Run Tests
test_losses = []

for width in widths:
    print(f"\nTesting width {width}...")

    # Recreate your model architecture
    model = resnet18_scaled(width_mult=width, num_classes=30).to(device)

    # Load weights
    state = torch.load(model_files[width], map_location=device)
    model.load_state_dict(state)

    # Evaluate
    loss = test_model_loss(model, test_loader, device, criterion)
    test_losses.append(loss)

    print(f"Loss for width={width}: {loss:.4f}")


#Plotting
plt.figure(figsize=(6,4))
plt.plot(widths, test_losses, marker='o')
plt.title("Test Loss vs Model Width")
plt.xlabel("Width")
plt.ylabel("Test Loss")
plt.show()
plt.savefig("test_loss_vs_width.png")

# Print losses
print("\nTest Losses:")
for width, loss in zip(widths, test_losses):
    print(f"Width {width}: Loss = {loss:.4f}")