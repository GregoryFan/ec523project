#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 19:32:08 2025

@author: jindongfeng
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from tqdm import tqdm

# ------------------------------
# 1) Paths
# ------------------------------
dataset_path = "/projectnb/ec523bn/students/roanfeng/dataset"
train_dir = os.path.join(dataset_path, "Train_Set_Folder")
val_dir = os.path.join(dataset_path, "Validation_Set_Folder")
test_dir = os.path.join(dataset_path, "Test_Set_Folder")

vit_path = "/projectnb/ec523bn/students/roanfeng/ViT_S16"  # Path to your ViT S16 model

save_path = dataset_path  # Where loss vectors will be saved

# ------------------------------
# 2) Device
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 3) Data transforms & loaders
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT usually uses 224x224
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ------------------------------
# 4) Load ViT S16
# ------------------------------
# Assume ViT_S16.py contains a class ViT_S16
from vit_model import ViT_S16  # replace with your model class name

# Standard ViT S16 initialization
num_classes = len(train_dataset.classes)
model = ViT_S16(embed_dim=256, num_classes=num_classes).to(device)  # default embedding 256

# Optionally load checkpoint if available
checkpoint_file = os.path.join(vit_path, "checkpoint.pth")  # replace with actual filename
if os.path.exists(checkpoint_file):
    model.load_state_dict(torch.load(checkpoint_file, map_location=device))

# ------------------------------
# 5) Loss and optimizer
# ------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# ------------------------------
# 6) Training & testing for 500 epochs with checkpoints and progress bar
# ------------------------------

training_loss_record = []
testing_loss_record = []

num_epochs = 500
checkpoint_dir = os.path.join(save_path, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(num_epochs):
    # ---------------- Train ----------------
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        train_bar.set_postfix({'Batch Loss': f'{loss.item():.4f}'})
    epoch_train_loss = running_loss / len(train_loader.dataset)
    training_loss_record.append(epoch_train_loss)
    
    # ---------------- Test ----------------
    model.eval()
    running_loss = 0.0
    test_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} Testing", leave=False)
    with torch.no_grad():
        for images, labels in test_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            test_bar.set_postfix({'Batch Loss': f'{loss.item():.4f}'})
    epoch_test_loss = running_loss / len(test_loader.dataset)
    testing_loss_record.append(epoch_test_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f} Test Loss: {epoch_test_loss:.4f}")
    
    # ---------------- Save checkpoint every 2 epochs ----------------
    if (epoch + 1) % 2 == 0:
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': training_loss_record,
            'test_loss': testing_loss_record
        }, checkpoint_file)
        print(f"Checkpoint saved: {checkpoint_file}")

# ------------------------------
# Save training/testing vectors at the end
# ------------------------------
np.save(os.path.join(save_path, "training_loss_record.npy"), np.array(training_loss_record))
np.save(os.path.join(save_path, "testing_loss_record.npy"), np.array(testing_loss_record))


# ------------------------------
# 7) Embedding dimension sweep on last epoch
# ------------------------------
embedding_dims = [8, 16, 32, 64, 128, 256, 512]

train_loss_sweep = []
test_loss_sweep = []

for dim in embedding_dims:
    print(f"Testing embedding dimension: {dim}")
    model = ViT_S16(embed_dim=dim, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train one epoch
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    train_loss_sweep.append(running_loss / len(train_loader.dataset))
    
    # Test one epoch
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    test_loss_sweep.append(running_loss / len(test_loader.dataset))

# Save sweep results
np.save(os.path.join(save_path, "train_loss_sweep.npy"), np.array(train_loss_sweep))
np.save(os.path.join(save_path, "test_loss_sweep.npy"), np.array(test_loss_sweep))

print("Training complete. Loss vectors saved to:", save_path)




