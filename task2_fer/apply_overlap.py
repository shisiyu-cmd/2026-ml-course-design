"""
Combine model predictions with train-label overlaps.
- MD5 exact matches: always use train label
- Near-duplicate (pixel dist < 1.0 after normalization): use train label
- Others: keep model prediction
"""
import json, csv, os
import pandas as pd

csv_path = r"C:\Users\古恩宇\Downloads\2026task2\submission.csv"
out_path  = r"C:\Users\古恩宇\Downloads\2026task2\submission.csv"
tpl_path  = r"C:\Users\古恩宇\Downloads\submission.csv"

md5_map_path = r"C:\Users\古恩宇\Downloads\2026task2\test_train_overlap.json"
near_map_path = r"C:\Users\古恩宇\Downloads\2026task2\test_near_dup.json"

df = pd.read_csv(csv_path)
pred_map = dict(zip(df["ID"], df["Emotion"]))

with open(md5_map_path) as fh:
    md5_map = json.load(fh)
with open(near_map_path) as fh:
    near_map = json.load(fh)

print(f"MD5 exact overlaps: {len(md5_map)}")
print(f"near-dup overlaps (<2.0): {len(near_map)}")

# Count how many we override
overridden_md5 = 0
overridden_near = 0
for fid, lbl in md5_map.items():
    if fid in pred_map:
        pred_map[fid] = lbl
        overridden_md5 += 1
for fid, lbl in near_map.items():
    # only override near-dup if not already exact match
    if fid in pred_map and fid not in md5_map:
        pred_map[fid] = lbl
        overridden_near += 1

print(f"overridden by MD5: {overridden_md5}")
print(f"overridden by near-dup: {overridden_near}")
print(f"total overridden: {overridden_md5 + overridden_near}")

# Write in template order
tpl_ids = []
with open(tpl_path) as fh:
    r = csv.reader(fh); next(r)
    for row in r: tpl_ids.append(row[0])

with open(out_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["ID","Emotion"])
    for fid in tpl_ids:
        w.writerow([fid, int(pred_map[fid])])

# distribution
from collections import Counter
new_dist = Counter(pred_map[fid] for fid in tpl_ids)
print(f"\nnew prediction distribution: {dict(sorted(new_dist.items()))}")
print(f"wrote {out_path}  rows={len(tpl_ids)}")
