"""Build cached arrays for fast training."""
import os, json
import numpy as np
from PIL import Image

BASE = r"C:\Users\古恩宇\Downloads\2026task2\fer_data\fer_data"
OUT = r"C:\Users\古恩宇\Downloads\2026task2\cache"
os.makedirs(OUT, exist_ok=True)
LABEL_MAP = {"Angry": 0, "Fear": 1, "Happy": 2, "Sad": 3, "Surprise": 4, "Neutral": 5}


def load_gray(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


tr = os.path.join(BASE, "train")
imgs, lbls, names = [], [], []
for folder, lbl in LABEL_MAP.items():
    d = os.path.join(tr, folder)
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            imgs.append(load_gray(os.path.join(d, f)))
            lbls.append(lbl)
            names.append(f"{folder}/{f}")
X = np.stack(imgs)
y = np.array(lbls, dtype=np.int64)
print("train", X.shape, y.shape, np.bincount(y))

te = os.path.join(BASE, "test")
tf = sorted([f for f in os.listdir(te) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
Xt = np.stack([load_gray(os.path.join(te, f)) for f in tf])
print("test", Xt.shape)

np.savez_compressed(os.path.join(OUT, "data.npz"), Xtr=X, ytr=y, Xte=Xt,
                    train_names=np.array(names), test_names=np.array(tf))
print("saved ->", OUT)
