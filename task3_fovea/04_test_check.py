# -*- coding: utf-8 -*-
"""检查测试集是否存在「中心凹不可见」的图像（决定是否需要输出 (0,0)）。

对每张测试图画 4 个面板：原画布 / CLAHE 增强 / 中央 512 放大 / 灰度反相，
便于人工判断黄斑区与中心凹暗斑是否可辨。
"""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

meta = C.load_meta()
cache = C.build_cache(meta)


def clahe_rgb(img, clip=3.0):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


keys = sorted([k for k in meta if k.startswith("test/")])
n = len(keys)
fig, axes = plt.subplots(n, 3, figsize=(15, 4.4 * n))
if n == 1:
    axes = axes[None, :]
for i, k in enumerate(keys):
    im = cache[k]
    S = im.shape[0]
    enh = clahe_rgb(im)
    c0, c1 = S // 2 - 300, S // 2 + 300
    crop = enh[c0:c1, c0:c1]
    axes[i, 0].imshow(im)
    axes[i, 0].set_title("%s 原始 (W=%d)" % (k, meta[k]["W"]), fontsize=9)
    axes[i, 1].imshow(enh)
    axes[i, 1].set_title("CLAHE", fontsize=9)
    axes[i, 2].imshow(crop)
    axes[i, 2].set_title("中央 600x600 放大", fontsize=9)
    for j in range(3):
        axes[i, j].axis("off")
plt.tight_layout()
out = os.path.join(C.HERE, "viz", "test_check.png")
plt.savefig(out, dpi=58)
plt.close()
print("已保存", out)

# ---- 定量指标：黄斑区(中央)与视网膜整体的对比度/清晰度 ----
print()
print("%-12s %8s %8s %8s %8s %8s" % ("key", "W", "中心亮度", "视网膜亮度", "中央std", "清晰度"))
rows = []
for k in keys:
    im = cache[k]
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    S = g.shape[0]
    h = int(S * 0.34)
    center = g[S // 2 - h:S // 2 + h, S // 2 - h:S // 2 + h]
    yy, xx = np.mgrid[0:S, 0:S]
    disk = ((xx - S / 2) ** 2 + (yy - S / 2) ** 2) < (S * 0.42) ** 2
    retina = g[disk]
    lap = cv2.Laplacian(g, cv2.CV_32F)
    sharp = float(lap[disk].std())
    rows.append((k, meta[k]["W"], center.mean(), retina.mean(), center.std(), sharp))
    print("%-12s %8d %8.1f %8.1f %8.1f %8.2f" % rows[-1])

arr = np.array([[r[2], r[3], r[4], r[5]] for r in rows])
print()
print("中心亮度最低的 3 张:", [rows[i][0] for i in np.argsort(arr[:, 0])[:3]])
print("中央std 最低的 3 张:", [rows[i][0] for i in np.argsort(arr[:, 2])[:3]])
print("清晰度最低的 3 张:", [rows[i][0] for i in np.argsort(arr[:, 3])[:3]])