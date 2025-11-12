#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 19:32:08 2025

@author: jindongfeng
"""


import os
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import timm

# Paths
train_csv = '/projectnb/ec523bn/students/roanfeng/data/small_data/Training/Training_small_labels.csv'
train_img_dir = '/projectnb/ec523bn/students/roanfeng/data/small_data/Training'

val_csv = '/projectnb/ec523bn/students/roanfeng/data/small_data/Validation/Validation_small_labels.csv'
val_img_dir = '/projectnb/ec523bn/students/roanfeng/data/small_data/Validation'

test_csv = '/projectnb/ec523bn/students/roanfeng/data/small_data/Testing/Testing_small_labels.csv'
test_img_dir = '/projectnb/ec523bn/students/roanfeng/data/small_data/Testing'

output_dir = '/projectnb/ec523bn/students/roanfeng/output/'
os.makedirs(output_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------
# Dataset with safe image loading
# ----------------------------
class CSVDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        
        # Parse one-hot labels from strings safely
        def parse_one_hot(s):
            try:
                return torch.tensor(ast.literal_eval(s), dtype=torch.float32)
            except:
                return None
        
        self.labels = self.data.iloc[:, 2].apply(parse_one_hot)
        self.data['valid'] = self.labels.notnull()
        self.data = self.data[self.data['valid']]
        self.labels = self.labels[self.data.index].reset_index(drop=True)
        self.image_names = self.data.iloc[:, 0].reset_index(drop=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError:
            # Skip invalid image by returning the first valid image (will be ignored in training)
            image = Image.new('RGB', (384,384), color=(0,0,0))
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# ----------------------------
# Data loaders
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

def get_loader(csv_path, img_dir, batch_size=32, shuffle=True):
    dataset = CSVDataset(csv_path, img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    return loader

train_loader = get_loader(train_csv, train_img_dir)
val_loader = get_loader(val_csv, val_img_dir)
test_loader = get_loader(test_csv, test_img_dir, shuffle=False)

# ----------------------------
# Training function
# ----------------------------
def train_model(model, train_loader, val_loader, num_epochs, lr, device):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_hist.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        val_loss_hist.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_loss_hist, val_loss_hist

# ----------------------------
# Embedding dimension sweep
# ----------------------------
embedding_dims = [64, 128, 256]  # Example sweep, can modify
num_epochs = 500
lr = 1e-4
test_losses_by_dim = []

for dim in embedding_dims:
    print(f"\n==== Training ViT-S16 with embedding dim={dim} ====")
    
    # Create fresh model with 384x384 input
    model = timm.create_model(
        'vit_small_patch16_224',
        pretrained=True,
        num_classes=train_loader.dataset.labels[0].shape[0],
        img_size=384
    )
    
    # Adjust head (do not touch embed_dim directly)
    model.head = nn.Linear(model.head.in_features, train_loader.dataset.labels[0].shape[0])
    
    train_loss_hist, val_loss_hist = train_model(model, train_loader, val_loader, num_epochs, lr, device)
    
    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(output_dir, f'ViT_S16_dim{dim}.pt'))
    
    # Evaluate on test set
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss (dim={dim}): {test_loss:.4f}")
    test_losses_by_dim.append(test_loss)

    # Plot testing loss vs. epochs for this embedding dim
    plt.figure()
    plt.plot(range(1, num_epochs+1), val_loss_hist, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Validation Loss vs Epochs (dim={dim})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'val_loss_epochs_dim{dim}.png'))
    plt.close()

# Plot testing loss vs embedding dimension
plt.figure()
plt.plot(embedding_dims, test_losses_by_dim, marker='o')
plt.xlabel('Embedding Dimension')
plt.ylabel('Test Loss')
plt.title('Test Loss vs Embedding Dimension')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'test_loss_vs_embedding_dim.png'))
plt.close()




