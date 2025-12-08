#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 19:07:59 2025

@author: jindongfeng
"""
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob

# --- 1. Configuration & Paths ---
TEST_DATA_PATH = "/Users/jindongfeng/Desktop/BU/EC523/test_data"
CHECKPOINT_DIR = "/Users/jindongfeng/Desktop/BU/EC523/checkpoints/" 
MODEL_DIR = "/Users/jindongfeng/Desktop/BU/EC523/"

# Target Specific Epoch
TARGET_EPOCH = "214"

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
PATCH_SIZE = 16
EMBED_DIM = 256
DEPTH = 4
NUM_HEADS = 8
MLP_DIM = 512

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# --- 2. Device Setup ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Device: MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using Device: CUDA")
else:
    device = torch.device("cpu")
    print("Using Device: CPU")

# --- 3. Dynamic Model Import ---
sys.path.append(MODEL_DIR)
try:
    from vit_model import ViT_S16
    print("Successfully imported ViT_S16.")
except ImportError:
    print(f"Error: Could not import vit_model.py from {MODEL_DIR}")
    sys.exit(1)

# --- 4. Data Preparation ---
data_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

if not os.path.exists(TEST_DATA_PATH):
    print(f"Error: Dataset path not found at {TEST_DATA_PATH}")
    sys.exit(1)

def is_valid_file(path):
    filename = os.path.basename(path)
    if filename.startswith("._") or filename.startswith("."):
        return False
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))

print("Loading Test Dataset...")
val_dataset = datasets.ImageFolder(
    root=TEST_DATA_PATH, 
    transform=data_transform,
    is_valid_file=is_valid_file
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = val_dataset.classes
num_classes = len(class_names)
print(f"Found {num_classes} classes.")

# --- 5. Helper Function: Evaluate ---
def evaluate_model(model, loader, device):
    """Runs a pass over the data and returns accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    if total == 0: return 0.0
    return 100 * correct / total

# --- 6. Load Specific Checkpoint (214) ---
print(f"\nSearching for checkpoint 214 in {CHECKPOINT_DIR}...")
checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{TARGET_EPOCH}*.pth"))

if not checkpoint_files:
    print(f"Error: No checkpoint found containing '{TARGET_EPOCH}'")
    sys.exit(1)

# Use the first match
target_cp_path = checkpoint_files[0]
print(f"Loading Checkpoint: {os.path.basename(target_cp_path)}")

# Initialize Model
model = ViT_S16(
    img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=3, num_classes=num_classes,
    embed_dim=EMBED_DIM, depth=DEPTH, num_heads=NUM_HEADS, mlp_dim=MLP_DIM, dropout=0.1
)
model.to(device)

# Load Weights
try:
    checkpoint = torch.load(target_cp_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Weights loaded successfully.")
except RuntimeError as e:
    print(f"Error loading weights: {e}")
    sys.exit(1)

# --- 7. Evaluate and Print Results ---
acc = evaluate_model(model, val_loader, device)

print("\n" + "="*50)
print(f"TARGET CHECKPOINT: {os.path.basename(target_cp_path)}")
print(f"ACCURACY ON TEST DATA: {acc:.2f}%")
print("="*50 + "\n")

# --- 8. Visualization ---
print("Generating visualization...")
model.eval()

correct_pool = []
wrong_pool = []
pool_limit = 50 # Collect up to 50 of each type before stopping scan

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        for i in range(len(labels)):
            img_tensor = images[i].cpu() 
            true_idx = labels[i].item()
            pred_idx = preds[i].item()
            
            sample = (img_tensor, true_idx, pred_idx)
            
            if true_idx != pred_idx:
                if len(wrong_pool) < pool_limit:
                    wrong_pool.append(sample)
            else:
                if len(correct_pool) < pool_limit:
                    correct_pool.append(sample)
        
        # Stop scanning if we have plenty of options for both
        if len(wrong_pool) >= pool_limit and len(correct_pool) >= pool_limit:
            break

print(f"Pool Collected: {len(correct_pool)} correct samples, {len(wrong_pool)} wrong samples.")

# Construct the plot batch (aim for 4 images total)
plot_list = []

# 1. Randomly pick at least one wrong sample (if available)
if wrong_pool:
    plot_list.append(random.choice(wrong_pool))

# 2. Randomly pick at least one correct sample (if available)
if correct_pool:
    plot_list.append(random.choice(correct_pool))

# 3. Fill the remaining 2 slots randomly from whatever is left
remaining_pool = correct_pool + wrong_pool
# Remove the ones we already picked (to avoid duplicates, though unlikely)
# A simple way to fill the rest is just random.choice from the combined lists
while len(plot_list) < 4 and remaining_pool:
    plot_list.append(random.choice(remaining_pool))

# Plotting Function
def imshow(img, ax):
    img = img.permute(1, 2, 0).numpy()
    img = img * STD + MEAN
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.axis('off')

if not plot_list:
    print("Could not collect any samples. Is dataset empty?")
else:
    fig, axes = plt.subplots(1, len(plot_list), figsize=(12, 5))
    if len(plot_list) == 1: axes = [axes] # Handle single image case
    
    fig.suptitle(f"Predictions from Epoch {TARGET_EPOCH} (Random Selection)", fontsize=14)

    for i, (img, true_idx, pred_idx) in enumerate(plot_list):
        true_label = class_names[true_idx]
        pred_label = class_names[pred_idx]
        
        color = 'green' if true_idx == pred_idx else 'red'
        
        imshow(img, axes[i])
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontweight='bold')

    plt.tight_layout()
    plt.show()