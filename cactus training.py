#!/usr/bin/env python
# coding: utf-8

# In[144]:


import pandas as pd
import os
import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split


# In[145]:


import torch
torch.backends.cudnn.benchmark = True


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[146]:


DATA_ROOT = "Cactus Dataset"
GRADES_DIR = os.path.join(DATA_ROOT, "Grades")
IMAGES_DIR = os.path.join(DATA_ROOT, "Images Dataset")

VIEWS = ["A4C", "PL", "PSAV", "PSMV"]


# In[147]:


def grade_to_class(grade):
    if grade <= 4:
        return 0  # Bad
    elif grade <= 6:
        return 1  # Okay
    else:
        return 2  # Good


# In[148]:


dfs = []
for csv_file in glob.glob(os.path.join(GRADES_DIR, "*.csv")):
    df = pd.read_csv(csv_file)
    dfs.append(df)
grades_df = pd.concat(dfs, ignore_index=True)
grades_df.head()


# In[149]:


#sanity check
grades_df["Grade"].describe()


# In[150]:


records = []
for _, row in grades_df.iterrows():
    view = row["Subfolder Name"]
    img_name = row["Image Name"]
    grade = row["Grade"]
    img_path = os.path.join(IMAGES_DIR, view, img_name)
    if os.path.exists(img_path):
        records.append({
            "path": img_path,
            "grade": grade,
            "view": view })

data_df = pd.DataFrame(records)
print("Total matched images:", len(data_df))
data_df.head()


# In[151]:


data_df = data_df[data_df["grade"] > 0].reset_index(drop=True)


# In[152]:


train_df, val_df = train_test_split(
    data_df,
    test_size=0.2,
    random_state=42,
    stratify=data_df["view"])


# In[153]:


class CactusDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = Image.open(row["path"]).convert("RGB")
        grade = torch.tensor(row["grade"], dtype=torch.float32)
        quality = torch.tensor(
            grade_to_class(row["grade"]),
            dtype=torch.long        )
        if self.transform:
            img = self.transform(img)
        return img, grade, quality


# In[154]:


train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)])

video_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)])


# In[155]:


train_ds = CactusDataset(train_df, train_tfms)
val_ds   = CactusDataset(val_df, val_tfms)

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=8,
    pin_memory=True)

val_loader = DataLoader(
    val_ds,
    batch_size=64,
    num_workers=8,
    pin_memory=True)


# In[156]:


from torchvision.models import resnet18, ResNet18_Weights

class QualityGradeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.grade_head = nn.Linear(feat_dim, 1)
        self.quality_head = nn.Linear(feat_dim, 3)

    def forward(self, x):
        features = self.backbone(x)
        grade = self.grade_head(features)
        quality_logits = self.quality_head(features)
        return grade, quality_logits


# In[157]:


model = QualityGradeModel().to(device)
# Freeze backbone
for param in model.backbone.parameters():
    param.requires_grad = False
grade_loss_fn = nn.SmoothL1Loss(beta=1.0)
quality_loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# In[158]:


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    with torch.set_grad_enabled(train):
        for x, grade, quality in loader:
            x = x.to(device)
            grade = grade.to(device).unsqueeze(1)
            quality = quality.to(device)
            pred_grade, pred_quality = model(x)
            loss_grade = grade_loss_fn(pred_grade, grade)
            loss_quality = quality_loss_fn(pred_quality, quality)
            loss = loss_grade + 0.5 * loss_quality
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)


# In[159]:


#sanity check
x, grade, quality = next(iter(train_loader))
x = x.to(device)

out = model(x)
print(type(out), len(out))


# In[161]:


best_val = float("inf")

for epoch in tqdm(range(50), desc="Epochs"):
    train_loss = run_epoch(train_loader, train=True)
    val_loss   = run_epoch(val_loader, train=False)
    # Save best model only
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_cactus_model.pt")
        print(f"  ↳ Saved new best model (val={best_val:.3f})")
    print(f"Epoch {epoch:02d} | Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f}")


# In[162]:


model.load_state_dict(torch.load("best_cactus_model.pt", map_location=device))
model.eval()


# In[170]:


# grab one sample
import random
idx = random.randint(0, len(val_ds) - 1)
x, grade, quality = val_ds[idx]
x = x.unsqueeze(0).to(device)
with torch.no_grad():
    pred_grade, pred_quality = model(x)
print("True grade:", grade.item())
print("Pred grade:", pred_grade.item())
print("Pred quality probs:", torch.softmax(pred_quality[0], dim=0))


# In[171]:


QUALITY_NAMES = ["Bad", "Okay", "Good"]

def predict_with_confidence(image_path):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img = val_tfms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_grade, pred_quality = model(img)
        grade = float(torch.clamp(pred_grade, 1.0, 10.0).item())
        probs = torch.softmax(pred_quality, dim=1)
        confidence, cls = torch.max(probs, dim=1)
    label = QUALITY_NAMES[cls.item()]
    confidence = confidence.item()
    return f"{label} (Grade: {round(grade)}) ({confidence:.2f} confidence)"


# In[172]:


for i in range(5):
    print(predict_with_confidence(data_df.iloc[i]["path"]))


# In[173]:


import torch.nn.functional as F

img_path = data_df.iloc[0]["path"]
img = Image.open(img_path).convert("RGB")
img = val_tfms(img).unsqueeze(0).to(device)

with torch.no_grad():
    g, q = model(img)
    probs = F.softmax(q, dim=1)

print("Pred grade:", g.item())
print("Class probs [Bad, Okay, Good]:", probs.cpu().numpy())


# In[174]:


torch.save(model.state_dict(), "cactus_pretrain_5epochs.pt")


# In[175]:


import cv2
def crop_center_wedge(img, top_frac=0.1, bottom_frac=0.9, side_frac=0.1):
    """
    Crop out UI and black borders.
    Assumes wedge is centered horizontally.
    """
    w, h = img.size
    left = int(w * side_frac)
    right = int(w * (1 - side_frac))
    top = int(h * top_frac)
    bottom = int(h * bottom_frac)
    return img.crop((left, top, right, bottom))

def sample_frames(video_path, every_n_frames=10, max_frames=50):
    """
    Returns a list of PIL Images sampled from a video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n_frames == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = crop_center_wedge(img)
            frames.append(img)

            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


# In[176]:


import torch.nn.functional as F

def predict_on_frames(frames):
    model.eval()
    grades = []
    quality_probs = []
    with torch.no_grad():
        for img in frames:
            x = video_tfms(img).unsqueeze(0).to(device)
            pred_grade, pred_quality = model(x)
            grades.append(pred_grade.item())
            quality_probs.append(F.softmax(pred_quality, dim=1).cpu().numpy()[0])
    return np.array(grades), np.array(quality_probs)


# In[177]:


QUALITY_LABELS = ["Bad", "Okay", "Good"]

def aggregate_video_prediction(grades, quality_probs):
    mean_grade = float(np.clip(grades.mean(), 1.0, 10.0))
    mean_quality_prob = quality_probs.mean(axis=0)
    quality_idx = mean_quality_prob.argmax()
    return {
        "grade": mean_grade,
        "quality": QUALITY_LABELS[quality_idx],
        "confidence": mean_quality_prob[quality_idx]
    }


# In[ ]:


def get_true_video_grade(video_path):
    """
    Given path to VideoX.mp4, load VideoX.csv and return the expert grade.
    """
    csv_path = video_path.replace(".mp4", ".csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    for col in ["grade", "Grade", "score", "Score"]:
        if col in df.columns:
            return float(df[col].mean())
    return None


# In[178]:


def predict_video(video_path):
    frames = sample_frames(video_path, every_n_frames=10, max_frames=50)
    if len(frames) == 0:
        raise ValueError("No frames extracted from video")
    grades, quality_probs = predict_on_frames(frames)
    result = aggregate_video_prediction(grades, quality_probs)
    true_grade = get_true_video_grade(video_path)
    print(f"Video: {os.path.basename(video_path)}")
    if true_grade is not None:
        print(f"True grade: {true_grade:.2f}")
    else:
        print("True grade: N/A")
    print(f"Pred grade: {result['grade']:.2f}")
    print(f"Quality: {result['quality']} ({result['confidence']:.2f} confidence)")
    return {
        "true_grade": true_grade,
        **result}


# In[179]:


predict_video("Cactus Dataset/Videos/Training/Video1.mp4")

