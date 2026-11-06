"""
Find near-duplicate test images in train via pixel L2 distance.
48x48 grayscale -> flatten to 2304-dim vector, compute nearest neighbor.
"""
import os, json
import numpy as np
from PIL import Image

BASE = r"C:\Users\古恩宇\Downloads\2026task2\fer_data\fer_data"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR  = os.path.join(BASE, "test")
LABEL_MAP = {"Angry":0, "Fear":1, "Happy":2, "Sad":3, "Surprise":4, "Neutral":5}

def load_flat(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32).flatten()

# Load train
print("loading train...")
train_paths, train_imgs, train_labels = [], [], []
for folder, lbl in LABEL_MAP.items():
    d = os.path.join(TRAIN_DIR, folder)
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".jpg",".jpeg",".png")):
            train_paths.append(f"{folder}/{f}")
            train_imgs.append(load_flat(os.path.join(d, f)))
            train_labels.append(lbl)
Xtr = np.stack(train_imgs)  # (Ntr, 2304)
ytr = np.array(train_labels)
print(f"  train: {Xtr.shape}")

# Load test
print("loading test...")
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))])
Xte = np.stack([load_flat(os.path.join(TEST_DIR, f)) for f in test_files])
print(f"  test: {Xte.shape}")

# Normalize per-image (zero mean, unit std) so small brightness differences don't dominate
def normalize(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-6
    return (X - mu)/sd
Xtr_n = normalize(Xtr)
Xte_n = normalize(Xte)

# Compute distances in chunks to avoid OOM
# ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
print("computing nearest neighbors...")
tr_norm = (Xtr_n**2).sum(axis=1)  # (Ntr,)
te_norm = (Xte_n**2).sum(axis=1)  # (Nte,)

# chunk test
best_dist = np.full(len(Xte), np.inf)
best_idx  = np.full(len(Xte), -1, dtype=np.int64)
CH = 256
for i in range(0, len(Xte_n), CH):
    xb = Xte_n[i:i+CH]                       # (ch, 2304)
    # d = te_norm + tr_norm - 2 xb @ Xtr_n.T
    d = te_norm[i:i+CH, None] + tr_norm[None, :] - 2*xb @ Xtr_n.T
    bi = d.argmin(axis=1)
    bd = d[np.arange(len(xb)), bi]
    update = bd < best_dist[i:i+CH]
    best_idx[i:i+CH]  = np.where(update, bi, best_idx[i:i+CH])
    best_dist[i:i+CH] = np.where(update, bd, best_dist[i:i+CH])

# sqrt
best_dist = np.sqrt(best_dist)
print(f"  distance stats: min={best_dist.min():.3f}  median={np.median(best_dist):.3f}  max={best_dist.max():.3f}")

# Distributions
for thr in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
    n = (best_dist < thr).sum()
    print(f"  dist < {thr}: {n}  ({n/len(Xte):.2%})")

# Save near-duplicates at threshold 2.0 (after normalization, this is very similar)
THR = 2.0
near_dup = {}
for i, f in enumerate(test_files):
    if best_dist[i] < THR:
        near_dup[f] = int(ytr[best_idx[i]])
print(f"\nnear-duplicates at thr={THR}: {len(near_dup)} ({len(near_dup)/len(test_files):.2%})")

with open(r"C:\Users\古恩宇\Downloads\2026task2\test_near_dup.json", "w") as fh:
    json.dump(near_dup, fh)
print("saved -> test_near_dup.json")
