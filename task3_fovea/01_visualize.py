# -*- coding: utf-8 -*-
"""可视化训练样本：原图 + GT 中心凹 + 检测框，并输出统计图。"""
import os, glob
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "2026task3", "detection")
OUT = os.path.join(HERE, "viz")
os.makedirs(OUT, exist_ok=True)

gt = pd.read_csv(os.path.join(ROOT, "fovea_localization_train_GT.csv"))
gt["key"] = gt["data"].astype(int)

boxes = {}
for p in sorted(glob.glob(os.path.join(ROOT, "train_location", "*.xml"))):
    r = ET.parse(p).getroot()
    bb = r.find("object").find("bndbox")
    k = int(os.path.splitext(os.path.basename(p))[0])
    boxes[k] = [float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]

# ---------- 1. 网格展示 ----------
sample_keys = [1, 2, 3, 4, 5, 6, 40, 41, 42, 80, 20, 60]
fig, axes = plt.subplots(3, 4, figsize=(22, 13))
for ax, k in zip(axes.ravel(), sample_keys):
    im = Image.open(os.path.join(ROOT, "train", "%04d.jpg" % k))
    W, H = im.size
    arr = np.asarray(im)
    ax.imshow(arr)
    row = gt[gt.key == k].iloc[0]
    x0, y0, x1, y1 = boxes[k]
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=1.5))
    ax.plot(row.Fovea_X, row.Fovea_Y, "r+", ms=14, mew=2)
    ax.set_title("#%d  %dx%d  GT=(%.0f,%.0f)" % (k, W, H, row.Fovea_X, row.Fovea_Y), fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "samples_grid.png"), dpi=70)
plt.close()

# ---------- 2. 中心凹局部放大（检查视觉特征） ----------
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, k in zip(axes.ravel(), [1, 2, 3, 5, 8, 12, 25, 44, 63, 77]):
    im = Image.open(os.path.join(ROOT, "train", "%04d.jpg" % k))
    row = gt[gt.key == k].iloc[0]
    cx, cy = row.Fovea_X, row.Fovea_Y
    half = 260
    L = int(max(0, cx - half)); T = int(max(0, cy - half))
    R = int(min(im.size[0], cx + half)); B = int(min(im.size[1], cy + half))
    crop = np.asarray(im.crop((L, T, R, B)))
    ax.imshow(crop)
    ax.plot(cx - L, cy - T, "r+", ms=16, mew=2)
    ax.set_title("#%d crop" % k, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fovea_crops.png"), dpi=80)
plt.close()

# ---------- 3. 坐标分布 ----------
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
ax = axes[0]
ax.scatter(gt.Fovea_X, gt.Fovea_Y, c=gt.key, cmap="viridis", s=28)
ax.invert_yaxis(); ax.set_xlabel("Fovea_X"); ax.set_ylabel("Fovea_Y")
ax.set_title("GT 绝对像素坐标 (注意尺寸混合)"); ax.grid(alpha=.3)

sizes = {}
for k in gt.key:
    with Image.open(os.path.join(ROOT, "train", "%04d.jpg" % k)) as im:
        sizes[k] = im.size
gt["W"] = gt.key.map(lambda k: sizes[k][0])
gt["H"] = gt.key.map(lambda k: sizes[k][1])
gt["nx"] = gt.Fovea_X / gt.W
gt["ny"] = gt.Fovea_Y / gt.H
ax = axes[1]
sc = ax.scatter(gt.nx, gt.ny, c=gt.W, cmap="coolwarm", s=28)
ax.invert_yaxis(); ax.set_xlabel("nx"); ax.set_ylabel("ny")
ax.set_title("归一化坐标 (蓝=1956宽, 红=2992宽)"); ax.grid(alpha=.3)
plt.colorbar(sc, ax=ax)

ax = axes[2]
ax.hist(gt.nx, bins=20, alpha=.6, label="nx")
ax.hist(gt.ny, bins=20, alpha=.6, label="ny")
ax.set_title("归一化坐标分布"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "coord_dist.png"), dpi=90)
plt.close()

print("已保存:")
print("  ", os.path.join(OUT, "samples_grid.png"))
print("  ", os.path.join(OUT, "fovea_crops.png"))
print("  ", os.path.join(OUT, "coord_dist.png"))
print()
print("归一化坐标统计:")
print(gt[["nx", "ny"]].describe().to_string())
print()
print("按尺寸分组:")
print(gt.groupby(["W", "H"]).agg(n=("key", "size"),
                                 nx_std=("nx", "std"), ny_std=("ny", "std"),
                                 nx_min=("nx", "min"), nx_max=("nx", "max")).to_string())
print()
print("基线 RMSE (用训练集均值预测, 像素空间):")
for name, sub in [("全部", gt), ("2992x2000", gt[gt.W == 2992]), ("1956x1934", gt[gt.W == 1956])]:
    e = np.concatenate([sub.Fovea_X - sub.Fovea_X.mean(), sub.Fovea_Y - sub.Fovea_Y.mean()])
    print("   %-12s n=%2d  RMSE=%.1f px" % (name, len(sub), np.sqrt((e ** 2).mean())))