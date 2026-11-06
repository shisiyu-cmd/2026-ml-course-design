# -*- coding: utf-8 -*-
"""把测试集预测叠加到图像上做人工核查（黄斑应位于视盘颞侧、呈暗斑）。"""
import os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

p = argparse.ArgumentParser()
p.add_argument("--pred", default="runs/test_pred.csv")
p.add_argument("--out", default="viz/test_pred_overlay.png")
a = p.parse_args()

meta = C.load_meta()
cache = C.build_cache(meta)
df = pd.read_csv(a.pred)

fig, axes = plt.subplots(5, 4, figsize=(20, 25))
for ax, r in zip(axes.ravel(), df.itertuples()):
    k = r.key
    m = meta[k]
    im = cache[k]
    g = np.asarray(im.convert("L"), np.float32) if hasattr(im, "convert") else \
        np.asarray(im)
    if g.ndim == 3:
        g = g.mean(2)
    S = g.shape[0]
    # 以预测点为中心取 400x400 放大图，便于看清局部解剖结构
    cx, cy = int(r.canvas_x), int(r.canvas_y)
    h = 200
    x0, x1 = max(0, cx - h), min(S, cx + h)
    y0, y1 = max(0, cy - h), min(S, cy + h)
    crop = im[y0:y1, x0:x1]
    ax.imshow(crop)
    ax.plot(cx - x0, cy - y0, "+", c="lime", ms=26, mew=2.5)
    ax.plot(cx - x0, cy - y0, "o", mfc="none", mec="lime", ms=16, mew=1.5)
    ax.set_title("%s  peak=%.2f  距圆心=%.2fr" % (k, r.peak, r.dist_r), fontsize=10)
    ax.axis("off")
plt.suptitle("测试集预测放大核查 (绿十字 = 预测的中心凹位置)", fontsize=15)
plt.tight_layout()
plt.savefig(a.out, dpi=62)
plt.close()
print("已保存", a.out)

# 整图版本
fig, axes = plt.subplots(5, 4, figsize=(20, 22))
for ax, r in zip(axes.ravel(), df.itertuples()):
    k = r.key
    ax.imshow(cache[k])
    ax.plot(r.canvas_x, r.canvas_y, "+", c="lime", ms=20, mew=2.5)
    ax.set_title(k, fontsize=10)
    ax.axis("off")
plt.tight_layout()
out2 = a.out.replace(".png", "_full.png")
plt.savefig(out2, dpi=62)
plt.close()
print("已保存", out2)