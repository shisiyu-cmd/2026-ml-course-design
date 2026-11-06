import os
from PIL import Image
from collections import Counter

base = r"C:\Users\古恩宇\Downloads\2026task2\fer_data\fer_data"

# Label mapping per user:
# 0 Angry, 1 Fear, 2 Happy, 3 Sad, 4 Surprise, 5 Neutral
label_map = {"Angry":0, "Fear":1, "Happy":2, "Sad":3, "Surprise":4, "Neutral":5}

print("=== TRAIN ===")
train_dir = os.path.join(base, "train")
size_counter = Counter()
mode_counter = Counter()
total = 0
for folder, lbl in label_map.items():
    fp = os.path.join(train_dir, folder)
    files = [f for f in os.listdir(fp) if f.lower().endswith((".jpg",".jpeg",".png"))]
    # sample a few to get size
    for f in files[:20]:
        with Image.open(os.path.join(fp, f)) as im:
            size_counter[im.size] += 1
            mode_counter[im.mode] += 1
    print(f"  {folder:10s} -> label {lbl}  n={len(files)}")
    total += len(files)
print("  total train:", total)
print("  sampled sizes:", size_counter.most_common(5))
print("  sampled modes:", mode_counter.most_common(5))

print("=== TEST ===")
test_dir = os.path.join(base, "test")
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
size_counter2 = Counter()
mode_counter2 = Counter()
for f in test_files[:50]:
    with Image.open(os.path.join(test_dir, f)) as im:
        size_counter2[im.size] += 1
        mode_counter2[im.mode] += 1
print("  n test:", len(test_files))
print("  sampled sizes:", size_counter2.most_common(5))
print("  sampled modes:", mode_counter2.most_common(5))
print("  first 5 test files:", test_files[:5])
