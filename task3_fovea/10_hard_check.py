# -*- coding: utf-8 -*-
"""放大检查失败样本：判断真值本身是否合理、模型错在哪。"""
import os, argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

p = argparse.ArgumentParser()
p.add_argument("--tta", default="runs/tta_base200.npz")
p.add_argument("--variant", default="plain")
p.add_argument("--n", type=int, default=8)
p.add_argument("--out", default="viz/hard_cases.png")
a = p.parse_args()

d = np.load(a.tta, allow_pickle=True)
keys = list(d["keys"]); gt = d["gt"]; pred = d["soft_" + a.variant]
err = np.sqrt(((pred - gt) ** 2).sum(1))
order = np.argsort(-err)[:a.n]

meta = C.load_meta()
cache = C.build_cache(meta)


def clahe_rgb(img, clip=3.0):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


n = len(order)
fig, axes = plt.subplots(n, 2, figsize=(13, 6.0 * n))
if n == 1:
    axes = axes[None, :]
for r, i in enumerate(order):
    k = keys[i]
    m = meta[k]
    im = cache[k]
    S = im.shape[0]
    gx, gy = C.to_canvas(m, *gt[i])
    px, py = C.to_canvas(m, *pred[i])
    # 以真值为中心取 440 画布像素(= 约 880 原图像素)的放大图
    h = 220
    cx, cy = int(gx), int(gy)
    x0, x1 = max(0, cx - h), min(S, cx + h)
    y0, y1 = max(0, cy - h), min(S, cy + h)
    for c, view in enumerate([im, clahe_rgb(im)]):
        crop = view[y0:y1, x0:x1]
        ax = axes[r, c]
        ax.imshow(crop)
        ax.plot(gx - x0, gy - y0, "o", mfc="none", mec="lime", ms=22, mew=2.2)
        ax.plot(gx - x0, gy - y0, "+", c="lime", ms=15, mew=2.2)
        ax.plot(px - x0, py - y0, "x", c="red", ms=15, mew=2.6)
        if px - x0 < 0 or px - x0 > crop.shape[1] or py - y0 < 0 or py - y0 > crop.shape[0]:
            ax.set_xlabel("预测点在视野外", color="red")
        ax.set_title("%s  %s  err=%.0fpx  peak=%.2f"
                     % (k, "原图" if c == 0 else "CLAHE", err[i], float(d["peak_mean"][i])),
                     fontsize=10)
        ax.axis("off")
plt.suptitle("绿圈=真值  红十字=模型预测 (取真值附近 880x880 原图像素放大)", fontsize=14)
plt.tight_layout()
plt.savefig(a.out, dpi=75)
plt.close()
print("已保存", a.out)

print()
print("失败样本明细 (原图像素):")
for r, i in enumerate(order):
    k = keys[i]
    m = meta[k]
    print("  %s  真值=(%7.1f,%7.1f)  预测=(%7.1f,%7.1f)  误差=%6.1f  峰值=%.3f  距圆心 真=%.2fr 预=%.2fr"
          % (k, gt[i][0], gt[i][1], pred[i][0], pred[i][1], err[i], float(d["peak_mean"][i]),
             np.hypot(gt[i][0] - m["cx"], gt[i][1] - m["cy"]) / m["r"],
             np.hypot(pred[i][0] - m["cx"], pred[i][1] - m["cy"]) / m["r"]))