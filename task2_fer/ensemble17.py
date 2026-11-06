"""
17-model ensemble: 5 ResNet34 + 5 ResNet18 + 2 EfficientNet-B0 + 5 GoogLeNet.
TTA: original + horizontal flip.
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

all_logits = []

# ResNet34 folds (112x112)
for fold in range(1,6):
    ckpt = rf"C:\Users\古恩宇\Downloads\2026task2\folds_r34\fold{fold}.pt"
    m = models.resnet34(weights=None); m.fc = nn.Linear(m.fc.in_features,6)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m = m.to(DEVICE)
    all_logits.append(predict(m, tf112)); print(f"  r34 fold{fold} done", flush=True)
    del m; torch.cuda.empty_cache()

# ResNet18 folds (96x96)
for fold in range(1,6):
    ckpt = rf"C:\Users\古恩宇\Downloads\2026task2\folds\fold{fold}.pt"
    m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features,6)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m = m.to(DEVICE)
    all_logits.append(predict(m, tf96)); print(f"  r18 fold{fold} done", flush=True)
    del m; torch.cuda.empty_cache()

# EfficientNet folds (112x112)
for fold in [1,2]:
    ckpt = rf"C:\Users\古恩宇\Downloads\2026task2\folds_eff\fold{fold}.pt"
    m = models.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features,6)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m = m.to(DEVICE)
    all_logits.append(predict(m, tf112)); print(f"  eff fold{fold} done", flush=True)
    del m; torch.cuda.empty_cache()

# GoogLeNet folds (112x112) - aux_logits=False for inference
for fold in range(1,6):
    ckpt = rf"C:\Users\古恩宇\Downloads\2026task2\folds_googlenet\fold{fold}.pt"
    m = models.googlenet(weights=None, aux_logits=False)
    m.fc = nn.Linear(m.fc.in_features,6)
    # load state dict, ignoring aux heads
    sd = torch.load(ckpt, map_location=DEVICE)
    sd = {k:v for k,v in sd.items() if not k.startswith('aux')}
    m.load_state_dict(sd, strict=False)
    m = m.to(DEVICE)
    all_logits.append(predict(m, tf112)); print(f"  googlenet fold{fold} done", flush=True)
    del m; torch.cuda.empty_cache()

avg = torch.stack(all_logits,0).mean(0)
preds = avg.argmax(1).numpy().tolist()
print(f"\nensemble of {len(all_logits)} models", flush=True)
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
