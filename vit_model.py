#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 18:08:42 2025

@author: jindongfeng
"""

# vit_model.py
# ViT_S16.py
# vit_model.py
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=256):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Conv2d to extract patches and map to embedding dimension
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)           # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2)           # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)      # [B, num_patches, embed_dim]
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_dim=512, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [num_patches, batch, embed_dim] for nn.MultiheadAttention
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + self.dropout(attn_out)
        x2 = self.norm2(x)
        x = x + self.dropout(self.mlp(x2))
        return x

class ViT_S16(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=10,
                 embed_dim=256, depth=4, num_heads=8, mlp_dim=512, dropout=0.1):
        super(ViT_S16, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)               # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B,1,embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Transformer expects [seq_len, batch, embed_dim]
        x = x.transpose(0, 1)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.transpose(0, 1)  # [B, seq_len, embed_dim]
        x = self.norm(x)
        cls_output = x[:, 0]   # CLS token output
        out = self.head(cls_output)
        return out
