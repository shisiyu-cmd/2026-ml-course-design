"""
FER upgrade: ResNet18 (ImageNet pretrained) + 5-fold StratifiedKFold
             + MixUp + RandAugment + TTA logits ensemble.
Labels: 0=Angry, 1=Fear, 2=Happy, 3=Sad, 4=Surprise, 5=Neutral.
"""
import os, time, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from collections import Counter

BASE = r"C:\Users\古恩宇\Downloads\2026task2\fer_data\fer_data"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR  = os.path.join(BASE, "test")
OUT_CSV   = r"C:\Users\古恩宇\Downloads\2026task2\submission.csv"
TPL_CSV   = r"C:\Users\古恩宇\Downloads\submission.csv"
CKPT_DIR  = r"C:\Users\古恩宇\Downloads\2026task2\folds"
os.makedirs(CKPT_DIR, exist_ok=True)

LABEL_MAP = {"Angry":0, "Fear":1, "Happy":2, "Sad":3, "Surprise":4, "Neutral":5}
INV_MAP   = {v:k for k,v in LABEL_MAP.items()}

SEED = 42
BATCH = 64
EPOCHS = 22
LR = 3e-4
N_FOLDS = 5
IMG_SIZE = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---- load all images into memory (48x48 grayscale uint8) ----
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
imgs = np.stack(imgs)
lbls = np.array(lbls, dtype=np.int64)
print(f"  train: {imgs.shape}  ({time.time()-t0:.1f}s)", flush=True)

print("loading test images...", flush=True)
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))])
test_imgs = np.stack([load_gray(os.path.join(TEST_DIR, f)) for f in test_files])
print(f"  test: {test_imgs.shape}", flush=True)

# ---- transforms ----
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=2, magnitude=7),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

class NpyDataset(Dataset):
    def __init__(self, imgs, lbls, transform):
        self.imgs = imgs
        self.lbls = lbls
        self.tf = transform
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img = Image.fromarray(self.imgs[i]).convert("RGB")
        return self.tf(img), int(self.lbls[i])

# ---- model ----
def build_model():
    m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 6)
    return m.to(DEVICE)

# ---- MixUp ----
def mixup(x, y, alpha=0.2):
    if alpha <= 0 or random.random() > 0.5:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1-lam) * x[idx]
    return x_mix, y, y[idx], lam

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ---- 5-fold training ----
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_val_accs = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(imgs, lbls), 1):
    print(f"\n========== FOLD {fold}/{N_FOLDS}  train={len(tr_idx)}  val={len(va_idx)} ==========", flush=True)
    tr_ds = NpyDataset(imgs[tr_idx], lbls[tr_idx], train_tf)
    va_ds = NpyDataset(imgs[va_idx], lbls[va_idx], eval_tf)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = build_model()
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_acc = 0.0
    ckpt_path = os.path.join(CKPT_DIR, f"fold{fold}.pt")
    for ep in range(1, EPOCHS+1):
        model.train()
        tloss, tn = 0.0, 0
        for x, y in tr_ld:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            x, ya, yb, lam = mixup(x, y, alpha=0.2)
            logits = model(x)
            loss = lam*criterion(logits, ya) + (1-lam)*criterion(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tloss += loss.item()*x.size(0); tn += x.size(0)
        sched.step()

        # val
        model.eval()
        vc, vn = 0, 0
        with torch.no_grad():
            for x, y in va_ld:
                x = x.to(DEVICE); y = y.to(DEVICE)
                vc += (model(x).argmax(1)==y).sum().item(); vn += x.size(0)
        va_acc = vc/vn
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), ckpt_path)
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:02d}/{EPOCHS}  tr_loss {tloss/tn:.3f}  va_acc {va_acc:.4f}  best {best_acc:.4f}  lr {sched.get_last_lr()[0]:.2e}", flush=True)
    fold_val_accs.append(best_acc)
    print(f"  FOLD {fold} best val acc = {best_acc:.4f}  (saved -> {ckpt_path})", flush=True)

print(f"\n===== 5-fold val accs: {[round(a,4) for a in fold_val_accs]}  mean={np.mean(fold_val_accs):.4f} =====", flush=True)

# ---- ensemble inference on test (orig + hflip TTA) ----
print("\nensembling on test set...", flush=True)
test_ds = NpyDataset(test_imgs, np.zeros(len(test_imgs), dtype=np.int64), eval_tf)
test_ld = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

all_logits = []
for fold in range(1, N_FOLDS+1):
    model = build_model()
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"fold{fold}.pt"), map_location=DEVICE))
    model.eval()
    fold_logits = []
    with torch.no_grad():
        for x, _ in test_ld:
            x = x.to(DEVICE)
            x_flip = torch.flip(x, dims=[3])
            logits = 0.5*(model(x) + model(x_flip))
            fold_logits.append(logits.cpu())
    all_logits.append(torch.cat(fold_logits, 0))
    print(f"  fold {fold} done", flush=True)

# average logits across folds
avg_logits = torch.stack(all_logits, 0).mean(0)
preds = avg_logits.argmax(1).numpy().tolist()
print("pred distribution:", Counter(preds), flush=True)

# ---- write CSV in template order ----
tpl = []
with open(TPL_CSV) as fh:
    r = csv.reader(fh); next(r)
    for row in r: tpl.append(row[0])
pred_map = dict(zip(test_files, preds))
with open(OUT_CSV, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["ID","Emotion"])
    for fid in tpl:
        w.writerow([fid, int(pred_map[fid])])
print(f"\nwrote {OUT_CSV}  rows={len(tpl)}", flush=True)
