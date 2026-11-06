# -*- coding: utf-8 -*-
"""生成训练集贴图：GT(绿) 与 allExt 预测(红) 并列，用于排查 GT 错标。"""
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

META = C.load_meta()
GT = C.load_gt_map(C.load_gt())
KEYS = sorted(GT)
PRED = pd.read_csv("runs/test_pred_allExt.csv").set_index("key") if False else None
# 训练图的预测来自 OOF（用 long800 的 OOF npz）
z = np.load("runs/long800_oof.npz", allow_pickle=True)
oof = {str(k): p for k, p in zip(z["keys"], z["soft"])}


def sheet(keys, out, ncol=5):
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.6 * nrow))
    axes = np.atleast_2d(axes)
    for i, k in enumerate(keys):
        r, c = divmod(i, ncol)
        ax = axes[r, c]
        img = C.imread_u("%s/%s/%s.jpg" % (C.ROOT, k.split("/")[0], k.split("/")[1]))
        m = META[k]
        gx, gy = GT[k]
        px, py = oof[k]
        d = float(np.hypot(gx - px, gy - py))
        half = int(0.75 * m["r"])
        mx, my = (gx + px) / 2, (gy + py) / 2
        H, W = img.shape[:2]
        x0 = int(np.clip(mx - half, 0, max(0, W - 2 * half)))
        y0 = int(np.clip(my - half, 0, max(0, H - 2 * half)))
        sub = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
        ax.imshow(sub)
        ax.plot(gx - x0, gy - y0, "+", c="lime", ms=17, mew=3.4, label="GT")
        ax.plot(px - x0, py - y0, "x", c="red", ms=12, mew=3.0, label="OOF预测")
        ax.plot([gx - x0, px - x0], [gy - y0, py - y0], "-", c="white", lw=1.0, alpha=0.6)
        ax.set_title("%s  err=%.0f px" % (k.split("/")[1], d),
                     fontsize=10, color=("red" if d > 40 else "black"))
        ax.axis("off")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    for j in range(len(keys), nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=70)
    plt.close()
    print("已保存", out, "(%d 张)" % len(keys))


for i in range(4):
    sheet(KEYS[i * 20:(i + 1) * 20], "viz/train_sheet_%d.png" % (i + 1))
print()
errs = np.array([np.hypot(GT[k][0] - oof[k][0], GT[k][1] - oof[k][1]) for k in KEYS])
print("训练集 OOF 误差: 中位 %.1f  p90 %.1f  max %.1f" % (np.median(errs), np.percentile(errs, 90), errs.max()))
print("误差 >40px 的图（重点人工核查 GT 是否错标）:")
for k, e in sorted(zip(KEYS, errs), key=lambda x: -x[1])[:12]:
    if e > 40:
        print("   %-12s %.1f px" % (k, e))