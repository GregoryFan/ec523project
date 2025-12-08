#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 14:04:59 2025

@author: jindongfeng
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Change working directory
# -----------------------------
os.chdir("/Users/jindongfeng/Desktop/BU/EC523")
print("Current working directory:", os.getcwd())

# -----------------------------
# Load .npy files
# -----------------------------
training_loss_record = np.load("training_loss_record.npy")
testing_loss_record  = np.load("testing_loss_record.npy")
train_loss_sweep     = np.load("train_loss_sweep.npy")
test_loss_sweep      = np.load("test_loss_sweep.npy")

# -----------------------------
# Plot testing & training loss vs epochs
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(testing_loss_record, label='Testing Loss', marker='o')
plt.plot(training_loss_record, label='Training Loss', marker='x', alpha=0.7)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Training Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot testing loss vs embedding dimension
# -----------------------------
# Ensure test_loss_sweep is at least 1D
embedding_dims = [8, 16, 32, 64, 128, 256, 512]  # known dimensions

# Ensure test_loss_sweep is at least 1D
test_loss_sweep = np.atleast_1d(test_loss_sweep)

# If 2D, average over epochs (axis=0) to get loss per embedding dim
if test_loss_sweep.ndim > 1:
    mean_test_loss_per_dim = test_loss_sweep.mean(axis=0)
else:
    mean_test_loss_per_dim = test_loss_sweep

plt.figure(figsize=(8,5))
plt.plot(embedding_dims, mean_test_loss_per_dim, marker='o')
plt.xlabel("Embedding Dimension")
plt.ylabel("Average Testing Loss")
plt.title("Testing Loss vs Embedding Dimension")
plt.xscale('log', base=2)  # optional: use log scale for clarity
plt.xticks(embedding_dims, embedding_dims)
plt.grid(True)
plt.tight_layout()
plt.show()
