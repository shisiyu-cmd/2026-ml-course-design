# -*- coding: utf-8 -*-
"""对最关键的两张不确定测试图做高倍解剖学核查。

检查项:
  1) 视盘检测是否可信（检出半径是否落在训练集分布内）
  2) 各模型预测点到视盘的距离是否落在训练集观测范围内
  3) 图上画出 0.30r 的解剖学期望环
"""
import os, argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C
from importlib import import_module

sys_mod = import_module("19_disc_prior") if False else None
p = argparse.ArgumentParser()
p.add_argument("--keys", nargs="*", default=["test/0081", "test/0086", "test/0093"])
p.add_argument("--out", default="viz/critical_zoom.png")
a = p.parse_args()

meta = C.load_meta()
cache = C.build_cache(meta)
gt = C.load_gt(); gt_map = C.load_gt_map(gt)


def detect_disc(canvas):
    g = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY).astype(np.float32)
    S = g.shape[0]
    yy, xx = np.mgrid[0:S, 0:S]
    inside = ((xx - S / 2) ** 2 + (yy - S / 2) ** 2) < (S * 0.46) ** 2
    thr = np.percentile(g[inside], 97)
    m = ((g >= thr) & inside).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    area = st[i, cv2.CC_STAT_AREA]
    if area < 0.0005 * S * S:
        return None
    return float(cen[i][0]), float(cen[i][1]), float(np.sqrt(area / np.pi))


# 训练集 dr 分布（用于判定合理性）
D = np.load("runs/tta_long800.npz", allow_pickle=True)
keys_tr = list(D["keys"])
dr_tr = []
for i, k in enumerate(keys_tr):
    d = detect_disc(cache[k])
    if d is None:
        continue
    m = meta[k]
    fx, fy = C.to_canvas(m, *D["gt"][i])
    dr_tr.append(np.hypot(fx - d[0], fy - d[1]) / m["r"])
dr_tr = np.array(dr_tr)
LO, HI = np.percentile(dr_tr, [1, 99])
print("训练集 中心凹-视盘距离/r : mean=%.3f std=%.3f  p1=%.3f p99=%.3f 范围=[%.3f, %.3f]"
      % (dr_tr.mean(), dr_tr.std(), LO, HI, dr_tr.min(), dr_tr.max()))
print()

dfs = [(os.path.basename(f).replace("test_pred_", "").replace(".csv", ""),
        pd.read_csv(f).set_index("key"))
       for f in ["runs/test_pred_clahe1.csv", "runs/test_pred_long800.csv"]]

fig, axes = plt.subplots(len(a.keys), 3, figsize=(19, 6.4 * len(a.keys)))
if len(a.keys) == 1:
    axes = axes[None, :]
for r, k in enumerate(a.keys):
    m = meta[k]
    im = cache[k]
    S = im.shape[0]
    d = detect_disc(im)
    lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
    cl = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    enh = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    print("%s  视盘=(%.0f,%.0f) 检出半径=%.1f (训练集 0.101r=%.0f)" %
          (k, d[0], d[1], d[2], 0.101 * m["r"] * C.CANVAS / m["si"]))
    for nm, df in dfs:
        px, py = float(df.loc[k, "canvas_x"]), float(df.loc[k, "canvas_y"])
        dr = np.hypot(px - d[0], py - d[1]) / (m["r"] * C.CANVAS / m["si"])
        flag = "  <<< 超出训练范围" if (dr < LO or dr > HI) else ""
        print("     %-10s 预测=(%.0f,%.0f)  距视盘/disc_r = %.3f r%s" % (nm, px, py, dr, flag))
    print()
    # 三视图: 原图 / CLAHE / CLAHE 放大
    cx, cy = int(d[0]), int(d[1])
    for c, (view, zoom) in enumerate([(im, 1.0), (enh, 1.0), (enh, 1.9)]):
        ax = axes[r, c]
        if zoom > 1:
            h = int(S / 2 / zoom)
            x0, x1 = max(0, cx - h), min(S, cx + h)
            y0, y1 = max(0, cy - h), min(S, cy + h)
            sub = view[y0:y1, x0:x1]
            ox, oy = x0, y0
        else:
            sub = view; ox, oy = 0, 0
        ax.imshow(sub)
        # 视盘
        ax.add_patch(plt.Circle((d[0] - ox, d[1] - oy), d[2], fill=False, ec="yellow", lw=2))
        ax.plot(d[0] - ox, d[1] - oy, "+", c="yellow", ms=14, mew=2.4)
        # 0.30r 期望环
        rr = 0.30 * m["r"] * C.CANVAS / m["si"]
        ax.add_patch(plt.Circle((d[0] - ox, d[1] - oy), rr, fill=False,
                                ec="lime", lw=1.6, ls="--"))
        for (nm, df), col in zip(dfs, ["red", "deepskyblue"]):
            px, py = float(df.loc[k, "canvas_x"]), float(df.loc[k, "canvas_y"])
            if ox <= px < ox + sub.shape[1] and oy <= py < oy + sub.shape[0]:
                ax.plot(px - ox, py - oy, "x", c=col, ms=13, mew=2.6, label=nm)
        ax.set_title("%s %s" % (k, ["原图", "CLAHE", "CLAHE 放大"][c]), fontsize=9)
        ax.axis("off")
plt.suptitle("黄=视盘(含检出半径)  绿虚线=0.30r 解剖学期望环  红/蓝=模型预测", fontsize=14)
plt.tight_layout()
plt.savefig(a.out, dpi=80)
plt.close()
print("已保存", a.out)