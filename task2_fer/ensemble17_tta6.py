"""
17-model ensemble with 6-way TTA: original, hflip, +5deg, -5deg, scale1.05, scale0.95.
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

def make_tf(size):
    base = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize(mean,std)])
    return base

print("loading test images...", flush=True)
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg",".png"))])
test_imgs = []
for f in test_files:
    with Image.open(os.path.join(TEST_DIR,f)) as im:
        test_imgs.append(im.convert("RGB"))
print(f"  {len(test_imgs)} images", flush=True)

def tta_batch(imgs, tf, size):
    """Return list of tensors: orig, hflip, +5, -5, scale1.05, scale0.95"""
    results = []
    # original
    results.append(torch.stack([tf(im) for im in imgs]))
    # hflip
    results.append(torch.stack([tf(im.transpose(Image.FLIP_LEFT_RIGHT)) for im in imgs]))
    # +5 deg
    results.append(torch.stack([tf(im.rotate(5, resample=Image.BILINEAR, fillcolor=0)) for im in imgs]))
    # -5 deg
    results.append(torch.stack([tf(im.rotate(-5, resample=Image.BILINEAR, fillcolor=0)) for im in imgs]))
    # scale 1.05 (center crop equivalent: resize bigger then center crop)
    big = int(size * 1.05)
    tf_big = transforms.Compose([transforms.Resize((big,big)), transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(mean,std)])
    results.append(torch.stack([tf_big(im) for im in imgs]))
    # scale 0.95 (resize smaller then pad to size)
    small = int(size * 0.95)
    tf_small = transforms.Compose([transforms.Resize((small,small)), transforms.Pad((size-small)//2, fill=0), transforms.ToTensor(), transforms.Normalize(mean,std)])
    results.append(torch.stack([tf_small(im) for im in imgs]))
    return results

def predict(model, size, batch=32):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0,len(test_imgs),batch):
            chunk = test_imgs[i:i+batch]
            tta_list = tta_batch(chunk, make_tf(size), size)
            tta_avg = None
            for t in tta_list:
                t = t.to(DEVICE)
                logits = model(t)
                if tta_avg is None: tta_avg = logits
                else: tta_avg = tta_avg + logits
            tta_avg = tta_avg / len(tta_list)
            out.append(tta_avg.cpu())
    return torch.cat(out,0)

all_logits = []

# ResNet34 (112)
for fold in range(1,6):
    ckpt = rf"C:\Users\古恩宇\Downloads\2026task2\folds_r34\fold{fold}.pt"
    m = models.resnet34(weights=None); m.fc = nn.Linear(m.fc.in_features,6)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m = m.to(DEVICE)
    all_logits.append(predict(m, 112)); print(f"  r34 fold{fold} done", flush=True)
    del m; torch.cuda.empty_cache()

# ResNet18 (96)
for fold in range(1,6):
    ckpt = rf"C:\Users\古恩宇\Downloads\2026task2\folds\fold{fold}.pt"
    m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features,6)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m = m.to(DEVICE)
    all_logits.append(predict(m, 96)); print(f"  r18 fold{fold} done", flush=True)
    del m; torch.cuda.empty_cache()

# EfficientNet (112)
for fold in [1,2]:
    ckpt = rf"C:\Users\古恩宇\Downloads\2026task2\folds_eff\fold{fold}.pt"
    m = models.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features,6)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m = m.to(DEVICE)
    all_logits.append(predict(m, 112)); print(f"  eff fold{fold} done", flush=True)
    del m; torch.cuda.empty_cache()

# GoogLeNet (112)
for fold in range(1,6):
    ckpt = rf"C:\Users\古恩宇\Downloads\2026task2\folds_googlenet\fold{fold}.pt"
    m = models.googlenet(weights=None, aux_logits=False)
    m.fc = nn.Linear(m.fc.in_features,6)
    sd = torch.load(ckpt, map_location=DEVICE)
    sd = {k:v for k,v in sd.items() if not k.startswith('aux')}
    m.load_state_dict(sd, strict=False); m = m.to(DEVICE)
    all_logits.append(predict(m, 112)); print(f"  googlenet fold{fold} done", flush=True)
    del m; torch.cuda.empty_cache()

avg = torch.stack(all_logits,0).mean(0)
preds = avg.argmax(1).numpy().tolist()
print(f"\nensemble of {len(all_logits)} models, 6-way TTA", flush=True)
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
