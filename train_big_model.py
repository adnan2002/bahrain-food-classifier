import glob
import os
import random

import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

CLASSES = ["balaleet", "egg_tomato", "fish", "gaimat", "halwa", "karak",
           "liver", "ma3krona", "nakhaj", "samboosa", "tikka"]

DATA_DIR = "data"
FOLDER_TO_CLASS = {"eggs_tomato": "egg_tomato"}  # folder -> class name
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEED = 42
VAL_PER_CLASS = 6
PROBE_EPOCHS = 12
FINETUNE_EPOCHS = 6
UNFREEZE_LAST_BLOCKS = 2
FINETUNE_LR = 1e-5
BATCH_SIZE = 16

random.seed(SEED)
torch.manual_seed(SEED)


def load_images():
    train_crops, train_cls, val_crops, val_cls = [], [], [], []
    for cls in CLASSES:
        folder = next((f for f, c in FOLDER_TO_CLASS.items() if c == cls), cls)
        files = sorted(glob.glob(f"{DATA_DIR}/{folder}/*"))
        if not files:
            raise FileNotFoundError(f"no images for class {cls} in data/{folder}")
        random.shuffle(files)
        val_files = files[:VAL_PER_CLASS]
        train_files = files[VAL_PER_CLASS:]
        for f in train_files:
            train_crops.append(Image.open(f).convert("RGB"))
            train_cls.append(CLASSES.index(cls))
        for f in val_files:
            val_crops.append(Image.open(f).convert("RGB"))
            val_cls.append(CLASSES.index(cls))
    return train_crops, train_cls, val_crops, val_cls


train_crops, train_cls, val_crops, val_cls = load_images()
print("train images:", len(train_crops), "| val images:", len(val_crops))

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def prep(crops, labels, tf):
    X = torch.stack([tf(c) for c in crops])
    return X, torch.tensor(labels)


Xtr, ytr = prep(train_crops, train_cls, train_tf)
Xva, yva = prep(val_crops, val_cls, val_tf)
train_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(TensorDataset(Xva, yva), batch_size=BATCH_SIZE)

vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=241)
vit.load_state_dict(torch.load("food_model.pkl", map_location="cpu"))
print(f"loaded food_model.pkl ({sum(p.numel() for p in vit.parameters()) / 1e6:.1f} M params)")

vit.head = nn.Linear(vit.embed_dim, len(CLASSES))
print("replaced 241-class head with", len(CLASSES), "-class head")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit = vit.to(device)
print("device:", device)


def evaluate(model, dl):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in dl:
            correct += (model(xb.to(device)).argmax(1) == yb.to(device)).sum().item()
            total += len(yb)
    return correct / total


def train_stage(model, dl, val_dl, epochs, lr, unfreeze):
    for p in model.parameters():
        p.requires_grad = False
    if unfreeze:
        for p in model.blocks[-unfreeze:].parameters():
            p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(yb)
        sched.step()
        acc = evaluate(model, val_dl)
        best = max(best, acc)
        print(f"epoch {epoch:2d} | loss {total_loss / len(dl.dataset):.4f} | val acc {acc:.4f}")
    return best


print("--- linear probe (backbone frozen) ---")
best_acc = train_stage(vit, train_dl, val_dl, PROBE_EPOCHS, 1e-3, unfreeze=None)
if UNFREEZE_LAST_BLOCKS:
    print(f"--- unfreeze last {UNFREEZE_LAST_BLOCKS} blocks (lr {FINETUNE_LR}) ---")
    best_acc = max(best_acc, train_stage(vit, train_dl, val_dl, FINETUNE_EPOCHS, FINETUNE_LR, unfreeze=UNFREEZE_LAST_BLOCKS))
print(f"best val acc: {best_acc:.4f}")

torch.save({
    "state_dict": vit.state_dict(),
    "class_names": CLASSES,
    "val_acc": best_acc,
}, "models/big_model.pt")
print(f"saved models/big_model.pt ({os.path.getsize('models/big_model.pt') / 1e6:.1f} MB)")
