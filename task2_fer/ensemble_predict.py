"""
Ensemble inference: 5 ResNet18 (96x96) + 2 EfficientNet-B0 (112x112).
TTA: original + horizontal flip.
"""
import os, csv, time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights
from PIL import Image
from collections import Counter

BASE = r"C:\Users\古恩宇\Downloads\2026task2\fer_data\fer_data"
TEST_DIR = os.path.join(BASE, "test")
OUT_CSV  = r"C:\Users\古恩宇\Downloads\2026task2\submission.csv"
TPL_CSV  = r"C:\Users\古恩宇\Downloads\submission.csv"
RES_DIR  = r"C:\Users\古恩宇\Downloads\2026task2\folds"
EFF_DIR  = r"C:\Users\古恩宇\Downloads\2026task2\folds_eff"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
tf96  = transforms.Compose([transforms.Resize((96,96)),
                             transforms.ToTensor(), transforms.Normalize(mean,std)])
tf112 = transforms.Compose([transforms.Resize((112,112)),
                             transforms.ToTensor(), transforms.Normalize(mean,std)])

# load test images
print("loading test images...")
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))])
test_imgs = []
for f in test_files:
    with Image.open(os.path.join(TEST_DIR, f)) as im:
        test_imgs.append(im.convert("RGB"))
print(f"  {len(test_imgs)} images")

def predict_with_model(model, tf, batch=64):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(test_imgs), batch):
            chunk = test_imgs[i:i+batch]
            x  = torch.stack([tf(im) for im in chunk]).to(DEVICE)
            xf = torch.stack([tf(im.transpose(Image.FLIP_LEFT_RIGHT)) for im in chunk]).to(DEVICE)
            logits = 0.5*(model(x) + model(xf))
            all_logits.append(logits.cpu())
    return torch.cat(all_logits, 0)

# --- ResNet18 folds ---
all_logits = []
for fold in range(1, 6):
    ckpt = os.path.join(RES_DIR, f"fold{fold}.pt")
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 6)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m = m.to(DEVICE)
    lg = predict_with_model(m, tf96)
    all_logits.append(lg)
    print(f"  resnet fold{fold} done  shape={lg.shape}", flush=True)
    del m; torch.cuda.empty_cache()

# --- EfficientNet folds ---
for fold in [1, 2]:
    ckpt = os.path.join(EFF_DIR, f"fold{fold}.pt")
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 6)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m = m.to(DEVICE)
    lg = predict_with_model(m, tf112)
    all_logits.append(lg)
    print(f"  efficientnet fold{fold} done  shape={lg.shape}", flush=True)
    del m; torch.cuda.empty_cache()

# ensemble
avg = torch.stack(all_logits, 0).mean(0)
preds = avg.argmax(1).numpy().tolist()
print(f"\nensemble of {len(all_logits)} models")
print("pred distribution:", Counter(preds))

# write in template order
tpl_ids = []
with open(TPL_CSV) as fh:
    r = csv.reader(fh); next(r)
    for row in r: tpl_ids.append(row[0])
pred_map = dict(zip(test_files, preds))
with open(OUT_CSV, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["ID","Emotion"])
    for fid in tpl_ids:
        w.writerow([fid, int(pred_map[fid])])
print(f"wrote {OUT_CSV}  rows={len(tpl_ids)}")
