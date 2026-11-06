"""
Weighted ensemble: weight each model by validation accuracy.
17 models: 5 R34 + 5 R18 + 2 EffNet + 5 GoogLeNet.
Weights use exponential of (val_acc - baseline).
"""
import os, csv
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from collections import Counter

BASE = r"C:\Users\古恩宇\Downloads\2026task2\fer_data\fer_data"
TEST_DIR = os.path.join(BASE, "test")
OUT_CSV  = r"C:\Users\古恩宇\Downloads\2026task2\submission.csv"
TPL_CSV  = r"C:\Users\古恩宇\Downloads\submission.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
tf96  = transforms.Compose([transforms.Resize((96,96)), transforms.ToTensor(), transforms.Normalize(mean,std)])
tf112 = transforms.Compose([transforms.Resize((112,112)), transforms.ToTensor(), transforms.Normalize(mean,std)])

# Validation accuracies per fold (from training logs)
R34_ACC   = [0.6937, 0.7033, 0.6829, 0.6833, 0.6888]
R18_AVG   = 0.6798  # 5-fold average, individual folds ~0.68
EFF_AVG   = 0.690   # estimated
GNET_ACC  = [0.6976, 0.7050, 0.6937, 0.6842, 0.6905]

# Exponential weighting: higher val acc -> significantly higher weight
def weight(acc):
    return float(np.exp((acc - 0.68) * 25))

R34_W  = [weight(a) for a in R34_ACC]
R18_W  = [weight(R18_AVG)] * 5
EFF_W  = [weight(EFF_AVG)] * 2
GNET_W = [weight(a) for a in GNET_ACC]

print("Weights:", flush=True)
print(f"  R34:  {[round(w,3) for w in R34_W]}", flush=True)
print(f"  R18:  {[round(w,3) for w in R18_W]}", flush=True)
print(f"  Eff:  {[round(w,3) for w in EFF_W]}", flush=True)
print(f"  GNet: {[round(w,3) for w in GNET_W]}", flush=True)

print("loading test images...", flush=True)
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg",".png"))])
test_imgs = []
for f in test_files:
    with Image.open(os.path.join(TEST_DIR,f)) as im:
        test_imgs.append(im.convert("RGB"))
print(f"  {len(test_imgs)} images", flush=True)

def predict(model, tf, batch=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0,len(test_imgs),batch):
            chunk = test_imgs[i:i+batch]
            x  = torch.stack([tf(im) for im in chunk]).to(DEVICE)
            xf = torch.stack([tf(im.transpose(Image.FLIP_LEFT_RIGHT)) for im in chunk]).to(DEVICE)
            out.append((0.5*(model(x)+model(xf))).cpu())
    return torch.cat(out,0)

weighted_sum = None
total_weight = 0.0

def add(logits, w):
    global weighted_sum, total_weight
    if weighted_sum is None: weighted_sum = w * logits
    else: weighted_sum = weighted_sum + w * logits
    total_weight += w

# ResNet34
for fold in range(1,6):
    ckpt = rf"C:\Users\古恩宇\Downloads\2026task2\folds_r34\fold{fold}.pt"
    m = models.resnet34(weights=None); m.fc = nn.Linear(m.fc.in_features,6)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m = m.to(DEVICE)
    add(predict(m, tf112), R34_W[fold-1]); print(f"  r34 fold{fold} w={R34_W[fold-1]:.3f} done", flush=True)
    del m; torch.cuda.empty_cache()

# ResNet18
for fold in range(1,6):
    ckpt = rf"C:\Users\古恩宇\Downloads\2026task2\folds\fold{fold}.pt"
    m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features,6)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m = m.to(DEVICE)
    add(predict(m, tf96), R18_W[fold-1]); print(f"  r18 fold{fold} w={R18_W[fold-1]:.3f} done", flush=True)
    del m; torch.cuda.empty_cache()

# EfficientNet
for fold in [1,2]:
    ckpt = rf"C:\Users\古恩宇\Downloads\2026task2\folds_eff\fold{fold}.pt"
    m = models.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features,6)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m = m.to(DEVICE)
    add(predict(m, tf112), EFF_W[fold-1]); print(f"  eff fold{fold} w={EFF_W[fold-1]:.3f} done", flush=True)
    del m; torch.cuda.empty_cache()

# GoogLeNet
for fold in range(1,6):
    ckpt = rf"C:\Users\古恩宇\Downloads\2026task2\folds_googlenet\fold{fold}.pt"
    m = models.googlenet(weights=None, aux_logits=False)
    m.fc = nn.Linear(m.fc.in_features,6)
    sd = torch.load(ckpt, map_location=DEVICE)
    sd = {k:v for k,v in sd.items() if not k.startswith('aux')}
    m.load_state_dict(sd, strict=False); m = m.to(DEVICE)
    add(predict(m, tf112), GNET_W[fold-1]); print(f"  gnet fold{fold} w={GNET_W[fold-1]:.3f} done", flush=True)
    del m; torch.cuda.empty_cache()

avg = weighted_sum / total_weight
preds = avg.argmax(1).numpy().tolist()
print(f"\nweighted ensemble, total_weight={total_weight:.2f}", flush=True)
print("pred distribution:", Counter(preds), flush=True)

tpl_ids = []
with open(TPL_CSV) as fh:
    r=csv.reader(fh); next(r)
    for row in r: tpl_ids.append(row[0])
pred_map = dict(zip(test_files, preds))
with open(OUT_CSV,"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["ID","Emotion"])
    for fid in tpl_ids: w.writerow([fid, int(pred_map[fid])])
print(f"wrote {OUT_CSV} rows={len(tpl_ids)}", flush=True)
