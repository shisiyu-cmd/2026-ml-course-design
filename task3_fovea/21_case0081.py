# -*- coding: utf-8 -*-
"""把 test/0081 彻底看清：所有候选方案的预测位置 + 多种增强视图。"""
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

K = "test/0081"
meta = C.load_meta()
cache = C.build_cache(meta)
im = cache[K]
m = meta[K]
S = im.shape[0]

cands = {
    "S1 long800(已交)": "runs/test_pred_long800.csv",
    "S2 全量allA+allB": "runs/test_pred_alldata.csv",
    "S3 折模型7": "runs/test_pred_folds7.csv",
    "S4 9模型": "runs/test_pred_ens9.csv",
    "clahe1(200轮)": "runs/test_pred_clahe1.csv",
}
colors = ["red", "lime", "orange", "deepskyblue", "magenta"]
pts = {}
for (nm, f), col in zip(cands.items(), colors):
    d = pd.read_csv(f).set_index("key")
    pts[nm] = (float(d.loc[K, "canvas_x"]), float(d.loc[K, "canvas_y"]),
               float(d.loc[K, "peak"]), col)

print("%-22s %12s %12s %8s" % ("方案", "画布X", "画布Y", "峰值"))
for nm, (x, y, pk, _) in pts.items():
    print("%-22s %12.1f %12.1f %8.3f" % (nm, x, y, pk))

# 多视图
lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
views = {"原图": im}
for clip, tag in [(2.0, "CLAHE c2"), (5.0, "CLAHE c5")]:
    l2 = lab.copy()
    l2[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l2[:, :, 0])
    views[tag] = cv2.cvtColor(l2, cv2.COLOR_LAB2RGB)
g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
views["反相"] = cv2.cvtColor(np.stack([255 - g] * 3, -1).astype(np.uint8), cv2.COLOR_RGB2BGR)

# 以各方案中位数为中心裁剪
cx = int(np.median([p[0] for p in pts.values()]))
cy = int(np.median([p[1] for p in pts.values()]))
h = 210
x0, x1 = max(0, cx - h), min(S, cx + h)
y0, y1 = max(0, cy - h), min(S, cy + h)

n = len(views)
fig, axes = plt.subplots(2, n, figsize=(5.0 * n, 10))
for j, (tag, view) in enumerate(views.items()):
    for r in range(2):
        ax = axes[r, j]
        if r == 0:
            ax.imshow(view); ox, oy = 0, 0; title = tag
        else:
            ax.imshow(view[y0:y1, x0:x1]); ox, oy = x0, y0
            title = tag + " 放大"
        for nm, (x, y, pk, col) in pts.items():
            if ox <= x < ox + (S if r == 0 else x1 - x0) and \
               oy <= y < oy + (S if r == 0 else y1 - y0):
                ax.plot(x - ox, y - oy, "+", c=col, ms=17, mew=2.8, label=nm)
                ax.plot(x - ox, y - oy, "o", mfc="none", mec=col, ms=11, mew=1.4)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
axes[0, 0].legend(fontsize=6, loc="upper left", framealpha=0.8)
plt.suptitle("test/0081 各方案预测位置（分歧 172 px）", fontsize=14)
plt.tight_layout()
plt.savefig("viz/case0081.png", dpi=85)
plt.close()
print("\n已保存 viz/case0081.png")

# 定量: 各候选点到"全量模型"位置的距离
print()
b = np.array(pts["S2 全量allA+allB"][:2])
for nm, (x, y, pk, _) in pts.items():
    print("%-22s 距 S2 = %6.1f 画布px = %6.1f 原图px" % (nm, np.hypot(x - b[0], y - b[1]),
                                                     np.hypot(x - b[0], y - b[1]) * m["si"] / C.CANVAS))