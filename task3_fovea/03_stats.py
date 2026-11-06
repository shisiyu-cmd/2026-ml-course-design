# -*- coding: utf-8 -*-
"""验证预处理管线 + 计算画布归一化统计量。"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

meta = C.load_meta()
cache = C.build_cache(meta)
gt = C.load_gt()
gt_map = C.load_gt_map(gt)

# ---------- 统计量 ----------
acc = np.zeros(3); acc2 = np.zeros(3); n = 0
for k, img in cache.items():
    x = img.astype(np.float64) / 255.0
    acc += x.reshape(-1, 3).mean(0) * x[..., 0].size
    acc2 += (x ** 2).reshape(-1, 3).mean(0) * x[..., 0].size
    n += x[..., 0].size
mean = acc / n
std = np.sqrt(np.maximum(acc2 / n - mean ** 2, 1e-8))
print("画布均值 mean = (%.4f, %.4f, %.4f)" % tuple(mean))
print("画布标准差 std = (%.4f, %.4f, %.4f)" % tuple(std))
print()
print("请把 common.py 的 NORM_MEAN/NORM_STD 更新为上面两行。")

# ---------- 校验裁剪 & 坐标映射 ----------
print()
print("坐标映射往返校验 (画布 -> 原图):")
mx = 0.0
for k in list(meta.keys())[:10]:
    m = meta[k]
    if k not in gt_map:
        continue
    gx, gy = gt_map[k]
    cx, cy = C.to_canvas(m, gx, gy)
    rx, ry = C.to_orig(m, cx, cy)
    mx = max(mx, abs(rx - gx), abs(ry - gy))
print("  最大往返误差 = %.6f px" % mx)

print()
print("中心凹在画布上的位置 (归一化, 应为 ~0.49/0.53):")
vals = []
for k, (gx, gy) in gt_map.items():
    m = meta[k]
    cx, cy = C.to_canvas(m, gx, gy)
    vals.append((cx / C.CANVAS, cy / C.CANVAS))
vals = np.array(vals)
print("  x: mean=%.4f std=%.4f min=%.4f max=%.4f" % (vals[:, 0].mean(), vals[:, 0].std(), vals[:, 0].min(), vals[:, 0].max()))
print("  y: mean=%.4f std=%.4f min=%.4f max=%.4f" % (vals[:, 1].mean(), vals[:, 1].std(), vals[:, 1].min(), vals[:, 1].max()))

# ---------- 可视化画布 ----------
sel = ["train/0001", "train/0002", "train/0003", "train/0004", "train/0005",
       "train/0040", "train/0080", "test/0081", "test/0090", "test/0100"]
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
for ax, k in zip(axes.ravel(), sel):
    m = meta[k]
    img = cache[k].copy()
    ax.imshow(img)
    if k in gt_map:
        gx, gy = gt_map[k]
        cx, cy = C.to_canvas(m, gx, gy)
        ax.plot(cx, cy, "r+", ms=16, mew=2.2)
    ax.set_title("%s  canvas %d" % (k, C.CANVAS), fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(C.HERE, "viz", "canvas_check.png"), dpi=75)
plt.close()
print("\n已保存 viz/canvas_check.png")

# ---------- 可视化增强后的样本 ----------
import random as _r
ds = C.FoveaDataset(list(gt_map.keys()), cache, meta, gt_map, train=True, sigma=2, seed=1)
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
for ax in axes.ravel():
    s = ds[_r.randrange(len(ds))]
    im = s["img"].numpy().transpose(1, 2, 0) * np.array(C.NORM_STD) + np.array(C.NORM_MEAN)
    im = np.clip(im, 0, 1)
    ax.imshow(im)
    ax.plot(s["canvas"][0], s["canvas"][1], "r+", ms=16, mew=2.2)
    ax.set_title("%s" % s["key"], fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(C.HERE, "viz", "aug_check.png"), dpi=70)
plt.close()
print("已保存 viz/aug_check.png")