"""
FER classifier - in-memory version (fast).
6 classes: 0=Angry, 1=Fear, 2=Happy, 3=Sad, 4=Surprise, 5=Neutral.
Images 48x48 grayscale.
"""
import os, random, time, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import Counter

BASE = r"C:\Users\古恩宇\Downloads\2026task2\fer_data\fer_data"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR  = os.path.join(BASE, "test")
OUT_CSV   = r"C:\Users\古恩宇\Downloads\2026task2\submission.csv"
CKPT      = r"C:\Users\古恩宇\Downloads\2026task2\fer_cnn_best.pt"

LABEL_MAP = {"Angry":0, "Fear":1, "Happy":2, "Sad":3, "Surprise":4, "Neutral":5}
INV_MAP   = {v:k for k,v in LABEL_MAP.items()}

SEED = 42
BATCH = 128
EPOCHS = 30
LR = 1e-3
VAL_FRAC = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def load_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)

# ---- load all train images into memory ----
print("loading train images into memory...")
t0 = time.time()
train_imgs, train_lbls = [], []
for folder, lbl in LABEL_MAP.items():
    d = os.path.join(TRAIN_DIR, folder)
    files = sorted([f for f in os.listdir(d) if f.lower().endswith((".jpg",".jpeg",".png"))])
    for f in files:
        train_imgs.append(load_gray(os.path.join(d, f)))
        train_lbls.append(lbl)
train_imgs = np.stack(train_imgs)          # (N,48,48) uint8
train_lbls = np.array(train_lbls, dtype=np.int64)
print(f"  loaded train: {train_imgs.shape} in {time.time()-t0:.1f}s")

# stratified split
rng = np.random.RandomState(SEED)
train_idx, val_idx = [], []
for c in range(6):
    idx = np.where(train_lbls == c)[0]
    rng.shuffle(idx)
    n_val = int(len(idx)*VAL_FRAC)
    val_idx.extend(idx[:n_val])
    train_idx.extend(idx[n_val:])
train_idx = np.array(train_idx); rng.shuffle(train_idx)
val_idx   = np.array(val_idx);   rng.shuffle(val_idx)
print(f"  train={len(train_idx)}  val={len(val_idx)}")

class NpyDataset(Dataset):
    def __init__(self, imgs, lbls, augment=False):
        self.imgs = imgs
        self.lbls = lbls
        self.augment = augment
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img = self.imgs[i]                       # uint8 (48,48)
        if self.augment:
            # horizontal flip
            if random.random() < 0.5:
                img = img[:, ::-1]
            # small affine: translate + slight rotation via numpy
            # simple random shift
            max_shift = 4
            dx = random.randint(-max_shift, max_shift)
            dy = random.randint(-max_shift, max_shift)
            img = np.roll(img, (dy, dx), axis=(0,1))
        x = torch.from_numpy(np.ascontiguousarray(img)).float()/255.0
        x = (x - 0.5)/0.5                        # [-1,1]
        x = x.unsqueeze(0)                       # (1,48,48)
        return x, int(self.lbls[i])

train_ds = NpyDataset(train_imgs[train_idx], train_lbls[train_idx], augment=True)
val_ds   = NpyDataset(train_imgs[val_idx],   train_lbls[val_idx],   augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

# class weights
cnt = Counter(train_lbls.tolist())
counts = np.array([cnt[i] for i in range(6)], dtype=np.float64)
w = counts.sum()/counts
w = w/w.mean()
class_weights = torch.tensor(w, dtype=torch.float32, device=DEVICE)
print("  class weights:", w.round(3).tolist())

# ---- model ----
class ConvBN(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True))
    def forward(self,x): return self.b(x)

class FERCNN(nn.Module):
    def __init__(self, nc=6):
        super().__init__()
        self.feat = nn.Sequential(
            ConvBN(1,32), ConvBN(32,32), nn.MaxPool2d(2), nn.Dropout2d(0.25),
            ConvBN(32,64), ConvBN(64,64), nn.MaxPool2d(2), nn.Dropout2d(0.25),
            ConvBN(64,128), ConvBN(128,128), nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.head = nn.Sequential(
            nn.Linear(128*6*6, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, nc))
    def forward(self,x):
        return self.head(torch.flatten(self.feat(x), 1))

model = FERCNN().to(DEVICE)
print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  dev={DEVICE}")

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def run_epoch(loader, train=True):
    model.train(train)
    tl, tc, tn = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tl += loss.item()*x.size(0)
            tc += (logits.argmax(1)==y).sum().item()
            tn += x.size(0)
    return tl/tn, tc/tn

best = 0.0
for ep in range(1, EPOCHS+1):
    t0 = time.time()
    tr_l, tr_a = run_epoch(train_loader, True)
    va_l, va_a = run_epoch(val_loader, False)
    scheduler.step()
    dt = time.time()-t0
    print(f"ep {ep:02d}/{EPOCHS}  tr {tr_l:.3f}/{tr_a:.4f}  va {va_l:.3f}/{va_a:.4f}  {dt:.1f}s  lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
    if va_a > best:
        best = va_a
        torch.save(model.state_dict(), CKPT)
        print(f"   -> best saved ({best:.4f})", flush=True)

print(f"\nBest val acc: {best:.4f}")

# ---- load test images ----
print("loading test images...")
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))])
test_imgs = np.stack([load_gray(os.path.join(TEST_DIR, f)) for f in test_files])
print(f"  test: {test_imgs.shape}")

# ---- predict with TTA (orig + hflip) ----
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()
preds = []
with torch.no_grad():
    for i in range(0, len(test_imgs), BATCH):
        chunk = test_imgs[i:i+BATCH]
        x1 = torch.from_numpy(chunk).float().unsqueeze(1)/255.0
        x1 = (x1-0.5)/0.5
        x2 = torch.flip(x1, dims=[3])
        x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
        logits = 0.5*(model(x1)+model(x2))
        preds.extend(logits.argmax(1).cpu().numpy().tolist())

with open(OUT_CSV, "w", newline="") as fh:
    wcsv = csv.writer(fh)
    wcsv.writerow(["ID","Emotion"])
    for f, p in zip(test_files, preds):
        wcsv.writerow([f, int(p)])
print(f"wrote {OUT_CSV}  rows={len(preds)}")
print("pred distribution:", Counter(preds))
