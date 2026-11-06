"""
Validate the train/test near-duplicate override using a held-out protocol.

Idea: FER-2013 has known train/test leakage. Before trusting the override on the
test set, we must confirm that a "pixel-distance < T -> copy train label" rule is
actually accurate. We simulate it on TRAIN data:
  - split train into A / B (stratified)
  - for every image in B, find its nearest neighbour in A
  - check whether the NN's label matches B's true label, as a function of distance

If accuracy at small distance is very high (e.g. >95%), the override is safe.
"""
import os, json
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold

BASE = r"C:\Users\古恩宇\Downloads\2026task2\fer_data\fer_data"
TRAIN_DIR = os.path.join(BASE, "train")
LABEL_MAP = {"Angry": 0, "Fear": 1, "Happy": 2, "Sad": 3, "Surprise": 4, "Neutral": 5}


def load_flat(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("L"), dtype=np.float32).flatten()


print("loading train...", flush=True)
paths, imgs, lbls = [], [], []
for folder, lbl in LABEL_MAP.items():
    d = os.path.join(TRAIN_DIR, folder)
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(f"{folder}/{f}")
            imgs.append(load_flat(os.path.join(d, f)))
            lbls.append(lbl)
X = np.stack(imgs)
y = np.array(lbls)
print(f"  train: {X.shape}", flush=True)


def normalize(A):
    mu = A.mean(axis=1, keepdims=True)
    sd = A.std(axis=1, keepdims=True) + 1e-6
    return (A - mu) / sd


Xn = normalize(X)

# hold out 20% as "B" (mimics unseen test), rest as "A" (mimics train pool)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
_, b_idx = next(iter(skf.split(Xn, y)))
a_idx = np.setdiff1d(np.arange(len(Xn)), b_idx)
print(f"  A={len(a_idx)}  B={len(b_idx)}", flush=True)

A, B = Xn[a_idx], Xn[b_idx]
ya, yb = y[a_idx], y[b_idx]

# nearest neighbour of each B in A
an = (A ** 2).sum(axis=1)
best_d = np.full(len(B), np.inf)
best_i = np.full(len(B), -1, dtype=np.int64)
CH = 512
for i in range(0, len(B), CH):
    xb = B[i:i + CH]
    d = (xb ** 2).sum(axis=1)[:, None] + an[None, :] - 2 * xb @ A.T
    j = d.argmin(axis=1)
    best_i[i:i + CH] = j
    best_d[i:i + CH] = d[np.arange(len(xb)), j]
best_d = np.sqrt(np.maximum(best_d, 0))

nn_correct = (ya[best_i] == yb)
print("\n=== label agreement of nearest train neighbour vs true label ===")
print(f"{'dist<thr':>10} {'n':>7} {'agreement':>10}")
for thr in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]:
    m = best_d < thr
    if m.sum() == 0:
        print(f"{thr:>10.1f} {0:>7} {'-':>10}")
        continue
    print(f"{thr:>10.1f} {m.sum():>7} {nn_correct[m].mean():>9.2%}")

# How many of the *test* images sit below each threshold (from the saved json we
# only have thr=2.0; recompute quickly for test too).
print("\nrecomputing test distances...", flush=True)
TEST_DIR = os.path.join(BASE, "test")
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".png"))])
Xt = normalize(np.stack([load_flat(os.path.join(TEST_DIR, f)) for f in test_files]))

xn = (Xn ** 2).sum(axis=1)
td = np.full(len(Xt), np.inf)
ti = np.full(len(Xt), -1, dtype=np.int64)
for i in range(0, len(Xt), CH):
    xb = Xt[i:i + CH]
    d = (xb ** 2).sum(axis=1)[:, None] + xn[None, :] - 2 * xb @ Xn.T
    j = d.argmin(axis=1)
    ti[i:i + CH] = j
    td[i:i + CH] = d[np.arange(len(xb)), j]
td = np.sqrt(np.maximum(td, 0))

print("\n=== TEST images below each distance threshold ===")
print(f"{'dist<thr':>10} {'n_test':>8} {'pct':>7}")
for thr in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]:
    n = int((td < thr).sum())
    print(f"{thr:>10.1f} {n:>8} {n/len(td):>6.2%}")

np.save(r"C:\Users\古恩宇\Downloads\2026task2\_test_nn_dist.npy", td)
np.save(r"C:\Users\古恩宇\Downloads\2026task2\_test_nn_idx.npy", ti)
np.save(r"C:\Users\古恩宇\Downloads\2026task2\_test_nn_lbl.npy", y[ti])
with open(r"C:\Users\古恩宇\Downloads\2026task2\_test_files_order.json", "w") as fh:
    json.dump(test_files, fh)
print("\nsaved nn dist/idx/lbl for test")
