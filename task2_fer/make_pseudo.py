"""
Step 1: Generate pseudo labels with 12-model ensemble.
Save test logits and confidence distribution.
"""
import os, csv
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

BASE = r"C:\Users\古恩宇\Downloads\2026task2\fer_data\fer_data"
TEST_DIR = os.path.join(BASE, "test")
OUT_DIR  = r"C:\Users\古恩宇\Downloads\2026task2"
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
for fold in range(1,6):
    ckpt = rf"C:\Users\古恩宇\Downloads\2026task2\folds_r34\fold{fold}.pt"
    m = models.resnet34(weights=None); m.fc = nn.Linear(m.fc.in_features,6)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m = m.to(DEVICE)
    all_logits.append(predict(m, tf112)); print(f"  r34 fold{fold} done", flush=True)
    del m; torch.cuda.empty_cache()
for fold in range(1,6):
    ckpt = rf"C:\Users\古恩宇\Downloads\2026task2\folds\fold{fold}.pt"
    m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features,6)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m = m.to(DEVICE)
    all_logits.append(predict(m, tf96)); print(f"  r18 fold{fold} done", flush=True)
    del m; torch.cuda.empty_cache()
for fold in [1,2]:
    ckpt = rf"C:\Users\古恩宇\Downloads\2026task2\folds_eff\fold{fold}.pt"
    m = models.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features,6)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m = m.to(DEVICE)
    all_logits.append(predict(m, tf112)); print(f"  eff fold{fold} done", flush=True)
    del m; torch.cuda.empty_cache()

avg = torch.stack(all_logits,0).mean(0)
probs = torch.softmax(avg, dim=1)
conf, pred = probs.max(dim=1)
conf = conf.numpy(); pred = pred.numpy()

print(f"\nconfidence distribution:", flush=True)
for th in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]:
    n = (conf >= th).sum()
    print(f"  >={th:.2f}: {n:5d} ({n/len(conf):.1%})", flush=True)

# Save
np.save(os.path.join(OUT_DIR, "test_logits.npy"), avg.numpy())
np.save(os.path.join(OUT_DIR, "test_files.npy"), np.array(test_files))
np.save(os.path.join(OUT_DIR, "test_conf.npy"), conf)
np.save(os.path.join(OUT_DIR, "test_pred.npy"), pred)
print(f"\nsaved logits, conf, pred", flush=True)

# Also save test images as uint8 array for training
test_arr = np.stack([np.asarray(im.convert("L"), dtype=np.uint8) for im in test_imgs])
np.save(os.path.join(OUT_DIR, "test_arr.npy"), test_arr)
print(f"saved test images: {test_arr.shape}", flush=True)
