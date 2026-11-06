# -*- coding: utf-8 -*-
"""结构性检查：热力图目标(高斯)是否真的落在中心凹暗斑上？

这是 E0 的配套验证。如果目标本身偏了，再强的模型也学不对。
同时检查增强后关键点是否仍然跟随图像 content 移动。
"""
import os, argparse, random
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

p = argparse.ArgumentParser()
p.add_argument("--keys", nargs="*", default=["train/0001", "train/0003", "train/0005",
                                             "train/0040", "train/0062", "train/0079"])
p.add_argument("--sigma", type=float, default=2.0)
p.add_argument("--out", default="viz/target_check.png")
p.add_argument("--aug_out", default="viz/target_check_aug.png")
a = p.parse_args()

meta = C.load_meta()
cache = C.build_cache(meta)
gt = C.load_gt(); gt_map = C.load_gt_map(gt)

fig, axes = plt.subplots(2, len(a.keys), figsize=(4.2 * len(a.keys), 9))
for j, k in enumerate(a.keys):
    m = meta[k]
    im = cache[k]
    cx, cy = C.to_canvas(m, *gt_map[k])
    # 目标热力图
    tgt = C.make_heatmap_target(cx / C.STRIDE, cy / C.STRIDE,
                                C.CANVAS // C.STRIDE, C.CANVAS // C.STRIDE,
                                a.sigma, "cpu").numpy()
    # 全图 + 轮廓
    ax = axes[0, j]
    ax.imshow(im)
    up = cv2.resize(tgt, (C.CANVAS, C.CANVAS), interpolation=cv2.INTER_LINEAR)
    ax.contour(up, levels=[0.1, 0.5, 0.9], colors=["cyan", "yellow", "red"],
               linewidths=1.2, alpha=0.9)
    ax.plot(cx, cy, "+", c="lime", ms=16, mew=2.5)
    ax.set_title("%s 目标等高线" % k, fontsize=9)
    ax.axis("off")
    # 中心凹放大
    ax = axes[1, j]
    h = 150
    x0, x1 = max(0, int(cx) - h), min(C.CANVAS, int(cx) + h)
    y0, y1 = max(0, int(cy) - h), min(C.CANVAS, int(cy) + h)
    ax.imshow(im[y0:y1, x0:x1])
    up2 = up[y0:y1, x0:x1]
    ax.contour(up2, levels=[0.1, 0.5, 0.9], colors=["cyan", "yellow", "red"],
               linewidths=1.2, alpha=0.9)
    ax.plot(cx - x0, cy - y0, "+", c="lime", ms=18, mew=2.5)
    ax.set_title("放大 (青10% 黄50% 红90%)", fontsize=8)
    ax.axis("off")
plt.suptitle("热力图高斯目标 vs 中心凹: 目标中心(绿十字)应落在暗斑中心", fontsize=13)
plt.tight_layout()
plt.savefig(a.out, dpi=80)
plt.close()
print("已保存", a.out)

# ---- 增强一致性: 同一张图做不同增强, 目标点应始终跟随中心凹 ----
rng = random.Random(3)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
k = a.keys[0]
m = meta[k]
for ax in axes.ravel():
    img = cache[k]
    cx, cy = C.to_canvas(m, *gt_map[k])
    img2, (px, py), _ = C.affine_aug(img, (cx, cy), rng)
    img2 = C.photo_aug(img2, rng)
    h = 170
    x0, x1 = max(0, int(px) - h), min(C.CANVAS, int(px) + h)
    y0, y1 = max(0, int(py) - h), min(C.CANVAS, int(py) + h)
    ax.imshow(np.clip(img2[y0:y1, x0:x1], 0, 255).astype(np.uint8))
    ax.plot(px - x0, py - y0, "+", c="lime", ms=20, mew=2.5)
    ax.set_title("aug", fontsize=8)
    ax.axis("off")
plt.suptitle("增强一致性检查 (%s): 绿十字应始终在暗斑上" % k, fontsize=13)
plt.tight_layout()
plt.savefig(a.aug_out, dpi=75)
plt.close()
print("已保存", a.aug_out)