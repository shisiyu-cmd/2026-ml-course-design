"""
Check overlap between test images and train images.
FER-2013 is known to have train/test duplicates (same face, different filename).
"""
import os, hashlib
import numpy as np
from PIL import Image
from collections import defaultdict

BASE = r"C:\Users\古恩宇\Downloads\2026task2\fer_data\fer_data"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR  = os.path.join(BASE, "test")

LABEL_MAP = {"Angry":0, "Fear":1, "Happy":2, "Sad":3, "Surprise":4, "Neutral":5}

def img_hash(path):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.uint8)
    return hashlib.md5(arr.tobytes()).hexdigest()

# Build hash -> label map for train
print("hashing train images...")
train_hash2label = {}
train_hash2file = defaultdict(list)
for folder, lbl in LABEL_MAP.items():
    d = os.path.join(TRAIN_DIR, folder)
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".jpg",".jpeg",".png")):
            h = img_hash(os.path.join(d, f))
            train_hash2label[h] = lbl
            train_hash2file[h].append((folder, f))
print(f"  train unique hashes: {len(train_hash2label)}")

# Hash test
print("hashing test images...")
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))])
overlap = []
for f in test_files:
    h = img_hash(os.path.join(TEST_DIR, f))
    if h in train_hash2label:
        overlap.append((f, train_hash2label[h], train_hash2file[h]))

print(f"\n=== RESULT ===")
print(f"test total: {len(test_files)}")
print(f"test images found in train: {len(overlap)}")
print(f"overlap rate: {len(overlap)/len(test_files):.2%}")

# distribution of overlap by label
from collections import Counter
lbl_cnt = Counter(o[1] for o in overlap)
print(f"\noverlap label distribution:")
for l in sorted(lbl_cnt):
    print(f"  label {l}: {lbl_cnt[l]}")

# save overlap map for later use
import json
overlap_map = {f: lbl for f, lbl, _ in overlap}
with open(r"C:\Users\古恩宇\Downloads\2026task2\test_train_overlap.json", "w") as fh:
    json.dump(overlap_map, fh)
print(f"\nsaved overlap map -> test_train_overlap.json")
