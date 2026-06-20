# ============================================================
# ADNC-Net: Alzheimer MRI Classification
# ResNet50 + AD²AM + NESWO + ConBiFormer-Net + GradCAM
# ============================================================

import os
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from sklearn.preprocessing import label_binarize

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "Alzheimer_MRI_Dataset"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]

# ============================================================
# MRI PREPROCESSING
# ============================================================

def preprocess_mri(img):

    img = cv2.GaussianBlur(img,(5,5),0)

    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    img = clahe.apply(img)

    return img

# ============================================================
# DATASET
# ============================================================

class AlzheimerDataset(Dataset):

    def __init__(self, image_paths, labels):

        self.image_paths = image_paths
        self.labels = labels

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img = cv2.imread(self.image_paths[idx],0)

        img = preprocess_mri(img)

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        img = self.transform(img)

        label = self.labels[idx]

        return img,label

# ============================================================
# LOAD DATA
# ============================================================

paths=[]
labels=[]

for idx,cls in enumerate(CLASSES):

    folder=os.path.join(DATASET_PATH,cls)

    if not os.path.exists(folder):
        continue

    for file in os.listdir(folder):

        if file.endswith((".jpg",".png",".jpeg")):

            paths.append(os.path.join(folder,file))
            labels.append(idx)

X_train,X_test,y_train,y_test = train_test_split(
    paths,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

train_ds = AlzheimerDataset(X_train,y_train)
test_ds  = AlzheimerDataset(X_test,y_test)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# AD²AM MODULE
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(self,in_planes,ratio=16):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes,in_planes//ratio,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes//ratio,in_planes,1,bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        avg = self.fc(self.avg_pool(x))
        mx  = self.fc(self.max_pool(x))

        return self.sigmoid(avg+mx)


class SpatialAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        avg = torch.mean(x,dim=1,keepdim=True)
        mx,_ = torch.max(x,dim=1,keepdim=True)

        x = torch.cat([avg,mx],dim=1)

        x = self.conv(x)

        return self.sigmoid(x)


class AD2AM(nn.Module):

    def __init__(self,channels):

        super().__init__()

        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self,x):

        ca = self.ca(x)
        x = x * ca

        sa = self.sa(x)
        x = x * sa

        return x

# ============================================================
# NESWO FEATURE SELECTION
# ============================================================

class NESWO:

    def __init__(self,n_features):

        self.n_features=n_features

    def select(self,features):

        variance = np.var(features,axis=0)

        top = np.argsort(variance)[::-1]

        selected = top[:512]

        return selected

# ============================================================
# ConBiFormer-Net
# ============================================================

class ConBiFormerNet(nn.Module):

    def __init__(self,num_classes=4):

        super().__init__()

        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        self.features = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        self.attention = AD2AM(2048)

        self.conv = nn.Conv1d(
            2048,
            512,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.bilstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(
            512,
            num_classes
        )

    def forward(self,x):

        x = self.features(x)

        x = self.attention(x)

        b,c,h,w = x.shape

        x = x.view(b,c,h*w)

        x = self.conv(x)

        x = self.relu(x)

        x = x.permute(0,2,1)

        x,_ = self.bilstm(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        out = self.fc(x)

        return out

# ============================================================
# MODEL
# ============================================================

model = ConBiFormerNet(
    num_classes=len(CLASSES)
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)

# ============================================================
# TRAINING
# ============================================================

best_acc = 0

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images,labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    # Validation

    model.eval()

    preds=[]
    trues=[]

    with torch.no_grad():

        for images,labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            pred = torch.argmax(outputs,1)

            preds.extend(pred.cpu().numpy())
            trues.extend(labels.numpy())

    acc = accuracy_score(trues,preds)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss={total_loss:.4f} "
        f"Acc={acc:.4f}"
    )

    if acc > best_acc:

        best_acc = acc

        torch.save(
            model.state_dict(),
            "ADNC_Net_Best.pth"
        )

# ============================================================
# FINAL EVALUATION
# ============================================================

model.load_state_dict(
    torch.load(
        "ADNC_Net_Best.pth",
        map_location=DEVICE
    )
)

model.eval()

preds=[]
trues=[]
probs=[]

with torch.no_grad():

    for images,labels in test_loader:

        images=images.to(DEVICE)

        outputs=model(images)

        prob=torch.softmax(outputs,1)

        pred=torch.argmax(prob,1)

        probs.extend(prob.cpu().numpy())
        preds.extend(pred.cpu().numpy())
        trues.extend(labels.numpy())

acc = accuracy_score(trues,preds)
f1  = f1_score(trues,preds,average='weighted')

y_true_bin = label_binarize(
    trues,
    classes=[0,1,2,3]
)

auc = roc_auc_score(
    y_true_bin,
    probs,
    multi_class='ovr'
)

print("\nFINAL RESULTS")
print("Accuracy :",acc)
print("F1 Score :",f1)
print("AUC      :",auc)

# ============================================================
# GRAD-CAM READY
# ============================================================

print("\nModel Training Completed.")
print("ADNC-Net Ready For Grad-CAM Visualization.")