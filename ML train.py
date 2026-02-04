#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


# In[2]:


# Adjust if your paths differ
PANECHO_ROOT = "/home/users/joshua04/141/PanEcho"
DATA_ROOT = "/home/users/joshua04/141/data/raw/echonet_pediatric/A4C"

sys.path.append(PANECHO_ROOT)
sys.path.append(os.path.join(PANECHO_ROOT, "src"))


# In[3]:


from src.models import FrameTransformer


# In[4]:


transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),])


# In[5]:


def read_video_frames(path, n_frames=16):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(length - 1, 0), n_frames).astype(int)

    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(frame))
        i += 1
    cap.release()
    while len(frames) < n_frames:
        frames.append(frames[-1])

    return torch.stack(frames, dim=1)  # (3, T, 224, 224)


# In[6]:


class EchoNetVideoDataset(Dataset):
    def __init__(self, root, split="train", n_frames=16):
        df = pd.read_csv(os.path.join(root, "FileList.csv"))
        df = df[df["Split"] != 5] if split == "train" else df[df["Split"] == 5]

        self.df = df.reset_index(drop=True)
        self.root = root
        self.n_frames = n_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.root, "Videos", row["FileName"])
        x = read_video_frames(path, self.n_frames)
        y = torch.tensor(row["EF"], dtype=torch.float32)
        return x, y


# In[8]:


train_loader = DataLoader(
    EchoNetVideoDataset(DATA_ROOT, split="train"),
    batch_size=2,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=2)

val_loader = DataLoader(
    EchoNetVideoDataset(DATA_ROOT, split="val"),
    batch_size=2,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=2)


# In[ ]:


backbone = FrameTransformer(
    arch="convnext_tiny",
    n_heads=8,
    n_layers=2,
    transformer_dropout=0.1,
    pooling="mean",
    clip_len=16)

model = nn.Sequential(
    backbone,
    nn.Linear(768, 1)).to(device)

print(model)


# In[11]:


for p in model[0].parameters():
    p.requires_grad = False

for p in model[1].parameters():
    p.requires_grad = True

print("Trainable parameters:")
for name, p in model.named_parameters():
    if p.requires_grad:
        print(" ", name)


# In[12]:


criterion = nn.MSELoss()   # EF regression
optimizer = optim.AdamW(
    model[1].parameters(),
    lr=1e-3,
    weight_decay=1e-4)

use_amp = (device == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


# In[16]:


def run_one_epoch(model, loader, train=True):
    model.train(train)
    total_loss = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).view(-1, 1)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x)
            loss = criterion(pred, y)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / total


# In[20]:


import math
num_epochs = 50
best_val_loss = float("inf")
best_path = "/home/users/joshua04/141/results/ef_best.pt"

for epoch in range(num_epochs):
    # 🔓 Unfreeze backbone after 5 epochs
    if epoch == 5:
        print("Unfreezing backbone...")
        for p in model[0].parameters():
            p.requires_grad = True
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-5,          # lower LR for fine-tuning
            weight_decay=1e-4)
    train_loss = run_one_epoch(model, train_loader, train=True)
    val_loss   = run_one_epoch(model, val_loader,   train=False)
    print(
        f"Epoch {epoch:02d} | "
        f"train MSE: {train_loss:.4f} | "
        f"val MSE: {val_loss:.4f}")
    # 💾 Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_path)
        print(f"  ✓ Saved new best model (val MSE = {val_loss:.4f})")
        
# In[ ]:


train_loss = run_one_epoch(model, train_loader, train=True)
val_loss   = run_one_epoch(model, val_loader,   train=False)

print(f"train loss: {train_loss:.3f}")
print(f"val   loss: {val_loss:.3f}")


# In[ ]:


save_path = "/home/users/joshua04/141/results/ef_finetuned.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

torch.save(model.state_dict(), save_path)
print("Saved to:", save_path)

