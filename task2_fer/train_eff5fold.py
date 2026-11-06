"""
EfficientNet-B0 (with built-in SE attention) 5-fold training.
Input 112x112, CutMix+MixUp+RandomErasing, 30 epochs.
Saved folds -> folds_eff/fold{i}.pt
"""
import os, time, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from collections import Counter

BASE = r"C:\Users\古恩宇\Downloads\2026task2\fer_data\fer_data"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR  = os.path.join(BASE, "test")
CKPT_DIR  = r"C:\Users\古恩宇\Downloads\2026task2\folds_eff"
os.makedirs(CKPT_DIR, exist_ok=True)

LABEL_MAP = {"Angry":0, "Fear":1, "Happy":2, "Sad":3, "Surprise":4, "Neutral":5}

SEED = 42
BATCH = 64
EPOCHS = 30
LR = 3e-4
N_FOLDS = 5
IMG_SIZE = 112
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def load_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)

print("loading train images...", flush=True)
t0 = time.time()
imgs, lbls = [], []
for folder, lbl in LABEL_MAP.items():
    d = os.path.join(TRAIN_DIR, folder)
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".jpg",".jpeg",".png")):
            imgs.append(load_gray(os.path.join(d, f)))
            lbls.append(lbl)
imgs = np.stack(imgs); lbls = np.array(lbls, dtype=np.int64)
print(f"  train: {imgs.shape}  ({time.time()-t0:.1f}s)", flush=True)

mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=2, magnitude=8),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

class NpyDS(Dataset):
    def __init__(self, imgs, lbls, tf):
        self.imgs = imgs; self.lbls = lbls; self.tf = tf
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.imgs[i]).convert("RGB")), int(self.lbls[i])

def build_model():
    m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 6)
    return m.to(DEVICE)

# ---- MixUp + CutMix ----
def mixup_cutmix(x, y):
    # 50% pure, 25% mixup, 25% cutmix
    r = random.random()
    if r < 0.5:
        return x, y, y, 1.0
    elif r < 0.75:
        lam = np.random.beta(0.3, 0.3)
        idx = torch.randperm(x.size(0), device=x.device)
        x_mix = lam*x + (1-lam)*x[idx]
        return x_mix, y, y[idx], lam
    else:
        # cutmix
        lam = np.random.beta(0.3, 0.3)
        idx = torch.randperm(x.size(0), device=x.device)
        H, W = x.size(2), x.size(3)
        cut = np.sqrt(1-lam)
        cw, ch = int(W*cut), int(H*cut)
        cx, cy = np.random.randint(0, W), np.random.randint(0, H)
        x1, x2 = max(0, cx-cw//2), min(W, cx+cw//2)
        y1, y2 = max(0, cy-ch//2), min(H, cy+ch//2)
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam_adj = 1 - (x2-x1)*(y2-y1)/(W*H)
        return x_mix, y, y[idx], lam_adj

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_accs = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(imgs, lbls), 1):
    print(f"\n===== FOLD {fold}/{N_FOLDS}  train={len(tr_idx)} val={len(va_idx)} =====", flush=True)
    tr_ld = DataLoader(NpyDS(imgs[tr_idx], lbls[tr_idx], train_tf),
                       batch_size=BATCH, shuffle=True, num_workers=0)
    va_ld = DataLoader(NpyDS(imgs[va_idx], lbls[va_idx], eval_tf),
                       batch_size=BATCH, shuffle=False, num_workers=0)
    model = build_model()
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best = 0.0
    ckpt = os.path.join(CKPT_DIR, f"fold{fold}.pt")
    for ep in range(1, EPOCHS+1):
        model.train()
        tl, tn = 0.0, 0
        for x, y in tr_ld:
            x = x.to(DEVICE); y = y.to(DEVICE)
            x, ya, yb, lam = mixup_cutmix(x, y)
            logits = model(x)
            loss = lam*criterion(logits, ya) + (1-lam)*criterion(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()*x.size(0); tn += x.size(0)
        sched.step()
        model.eval()
        vc, vn = 0, 0
        with torch.no_grad():
            for x, y in va_ld:
                x = x.to(DEVICE); y = y.to(DEVICE)
                vc += (model(x).argmax(1)==y).sum().item(); vn += x.size(0)
        acc = vc/vn
        if acc > best:
            best = acc
            torch.save(model.state_dict(), ckpt)
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:02d}/{EPOCHS}  tr_loss {tl/tn:.3f}  va {acc:.4f}  best {best:.4f}", flush=True)
    fold_accs.append(best)
    print(f"  FOLD {fold} best = {best:.4f}", flush=True)

print(f"\n===== EfficientNet folds: {[round(a,4) for a in fold_accs]}  mean={np.mean(fold_accs):.4f} =====", flush=True)
