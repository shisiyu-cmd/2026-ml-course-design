# -*- coding: utf-8 -*-
"""生成测试集贴图：每张图标注 v4 与 ext5 的预测位置，供人工判读。"""
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

META = C.load_meta()
KEYS = sorted([k for k in META if k.startswith("test/")])
V4 = pd.read_csv("runs/test_pred_alldata.csv").set_index("key")
E5 = pd.read_csv("runs/test_pred_ext5.csv").set_index("key")


def crop_center(img, cx, cy, half):
    H, W = img.shape[:2]
    x0 = int(np.clip(cx - half, 0, max(0, W - 2 * half)))
    y0 = int(np.clip(cy - half, 0, max(0, H - 2 * half)))
    return img[y0:y0 + 2 * half, x0:x0 + 2 * half], x0, y0


def sheet(keys, out, half=None, ncol=5):
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.7 * nrow))
    axes = np.atleast_2d(axes)
    for i, k in enumerate(keys):
        r, c = divmod(i, ncol)
        ax = axes[r, c]
        img = C.imread_u("%s/%s/%s.jpg" % (C.ROOT, k.split("/")[0], k.split("/")[1]))
        m = META[k]
        p4 = (V4.x[k], V4.y[k])
        p5 = (E5.x[k], E5.y[k])
        d = float(np.hypot(p4[0] - p5[0], p4[1] - p5[1]))
        # 以两预测的中点为裁剪中心，保证两者都在画面里
        mx, my = (p4[0] + p5[0]) / 2, (p4[1] + p5[1]) / 2
        h = half or int(0.75 * m["r"])
        sub, x0, y0 = crop_center(img, mx, my, h)
        ax.imshow(sub)
        ax.plot(p4[0] - x0, p4[1] - y0, "o", mfc="none", mec="lime", ms=13, mew=2.4,
                label="v4 (15.3)")
        ax.plot(p5[0] - x0, p5[1] - y0, "x", c="red", ms=12, mew=3.2, label="ext5 (12.87)")
        ax.plot([p4[0] - x0, p5[0] - x0], [p4[1] - y0, p5[1] - y0], "-", c="white",
                lw=1.0, alpha=0.7)
        ax.set_title("%s   两者相差 %.0f px" % (k.split("/")[1], d), fontsize=10)
        ax.axis("off")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    for j in range(len(keys), nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=72)
    plt.close()
    print("已保存", out)


sheet(KEYS[:10], "viz/test_sheet_A.png")
sheet(KEYS[10:], "viz/test_sheet_B.png")

# 同时打印数值，便于对照
print()
print("%-11s %16s %16s %8s %8s" % ("key", "v4", "ext5", "相差", "峰值ext5"))
d = pd.read_csv("runs/test_pred_ext5.csv").set_index("key")
for k in KEYS:
    print("%-11s %16s %16s %8.1f %8.3f"
          % (k, "(%.0f,%.0f)" % (V4.x[k], V4.y[k]), "(%.0f,%.0f)" % (E5.x[k], E5.y[k]),
             np.hypot(V4.x[k] - E5.x[k], V4.y[k] - E5.y[k]), d.peak[k]))