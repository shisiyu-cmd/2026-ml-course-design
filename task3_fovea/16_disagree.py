# -*- coding: utf-8 -*-
"""对分歧最大的测试图做放大核查：对比不同模型族给出的位置，人工判断哪个落在中心凹上。"""
import os, argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

p = argparse.ArgumentParser()
p.add_argument("--preds", nargs="+",
               default=["runs/test_pred_final.csv", "runs/test_pred_long800.csv"])
p.add_argument("--keys", nargs="*", default=["test/0081", "test/0086", "test/0091",
                                             "test/0096", "test/0099"])
p.add_argument("--out", default="viz/disagree_check.png")
p.add_argument("--zoom", type=int, default=260)
a = p.parse_args()

meta = C.load_meta()
cache = C.build_cache(meta)


def clahe_rgb(img, clip=3.0):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


dfs = [pd.read_csv(f).set_index("key") for f in a.preds]
names = [os.path.basename(f).replace("test_pred_", "").replace(".csv", "")
         for f in a.preds]
colors = ["red", "deepskyblue", "orange", "magenta"]

n = len(a.keys)
fig, axes = plt.subplots(n, 3, figsize=(16, 5.2 * n))
if n == 1:
    axes = axes[None, :]
for r, k in enumerate(a.keys):
    m = meta[k]
    im = cache[k]
    S = im.shape[0]
    enh = clahe_rgb(im)
    pts = [(nm, float(df.loc[k, "canvas_x"]), float(df.loc[k, "canvas_y"]))
           for df, nm in zip(dfs, names)]
    # 用各预测的中位数作为视图中心
    cx = int(np.median([q[1] for q in pts]))
    cy = int(np.median([q[2] for q in pts]))
    h = a.zoom
    x0, x1 = max(0, cx - h), min(S, cx + h)
    y0, y1 = max(0, cy - h), min(S, cy + h)
    for c, view in enumerate([im, enh, enh]):
        ax = axes[r, c]
        sc = 1.0 if c == 0 else (2.2 if c == 2 else 1.0)
        ax.imshow(view[y0:y1, x0:x1])
        for (nm, px, py), col in zip(pts, colors):
            if x0 <= px < x1 and y0 <= py < y1:
                ax.plot(px - x0, py - y0, "+", c=col, ms=20 * 1.0, mew=2.6,
                        label="%s (%.0f,%.0f)" % (nm, px, py))
        if c == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.75)
        ax.set_title("%s  %s" % (k, ["原图", "CLAHE", "CLAHE 放大2.2x"][c]), fontsize=9)
        ax.axis("off")
plt.suptitle("分歧测试图核查：各模型族给出的中心凹位置", fontsize=14)
plt.tight_layout()
plt.savefig(a.out, dpi=80)
plt.close()
print("已保存", a.out)
print()
for k in a.keys:
    print(k)
    for df, nm in zip(dfs, names):
        print("   %-22s 画布=(%6.1f,%6.1f)  peak=%.3f" %
              (nm, df.loc[k, "canvas_x"], df.loc[k, "canvas_y"], df.loc[k, "peak"]))
    print()