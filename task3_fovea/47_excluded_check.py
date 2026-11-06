# -*- coding: utf-8 -*-
"""检查被"离心度 >0.62r"规则剔除的 IDRiD 图像是否其实可用。"""
import os, io, csv, json, zipfile
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

ZIP = "ext/data/IDRiD_Localization.zip"
EXT = os.path.join(C.ROOT, "ext")
z = zipfile.ZipFile(ZIP)
names = z.namelist()

gt = {}
for sub in ["IDRiD_Fovea_Center_Training Set_Markups.csv"]:
    f = [n for n in names if sub in n][0]
    for row in csv.reader(io.StringIO(z.read(f).decode("utf-8", "replace"))):
        if len(row) < 3 or not row[0].strip().startswith("IDRiD_"):
            continue
        try:
            gt[row[0].strip()] = (float(row[1]), float(row[2]))
        except ValueError:
            pass

imgs = {}
for n in names:
    stem = os.path.splitext(os.path.basename(n))[0]
    if n.lower().endswith(".jpg") and "Training Set" in n and stem in gt:
        imgs[stem] = n

have = set(r["key"] for r in csv.DictReader(open(os.path.join(EXT, "gt.csv"))))
excluded = [s for s in sorted(imgs) if "ext/%s" % s.lower() not in have]
print("IDRiD 训练集共 %d 张；已采用 %d 张；被剔除 %d 张"
      % (len(imgs), len(imgs) - len(excluded), len(excluded)))

N = min(20, len(excluded))
sel = [excluded[i] for i in np.linspace(0, len(excluded) - 1, N).astype(int)]
fig, axes = plt.subplots(4, 5, figsize=(21, 18))
for i, stem in enumerate(sel):
    ax = axes[i // 5, i % 5]
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(imgs[stem]), np.uint8),
                                    cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    gx, gy = gt[stem]
    half = int(0.34 * min(H, W))
    x0 = int(np.clip(gx - half, 0, max(0, W - 2 * half)))
    y0 = int(np.clip(gy - half, 0, max(0, H - 2 * half)))
    ax.imshow(img[y0:y0 + 2 * half, x0:x0 + 2 * half])
    ax.plot(gx - x0, gy - y0, "+", c="lime", ms=17, mew=3.4)
    # 标出图像几何中心，看真值离中心多远
    ax.plot(W / 2 - x0, H / 2 - y0, "x", c="cyan", ms=11, mew=2.4)
    ax.set_title("%s  真值离图心 %.2fW" % (stem, np.hypot(gx - W/2, gy - H/2) / W), fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("viz/excluded_check.png", dpi=62)
print("已保存 viz/excluded_check.png")
print()
print("判读要点: 绿+ 是否落在中心凹（颞侧血管弓内的暗区）？")
print("          若多数看起来正确，说明 0.62r 阈值过严，应放宽。")