# -*- coding: utf-8 -*-
"""检验"周边误差源于光照衰减(暗角)"这一假设。

做法: 在 GT 中心凹附近统计局部亮度/对比度，与 OOF 误差求相关。
若暗角是主因，则误差应与局部亮度显著负相关 —— 那样"光照归一化"会比
"按离心度过采样"更对症。
"""
import numpy as np
import pandas as pd
import cv2
import common as C

META = C.load_meta()
gt = C.load_gt_map(C.load_gt())

rows = []
for tag in ["long800"]:
    z = np.load("runs/%s_oof.npz" % tag, allow_pickle=True)
    soft, orig, keys = z["soft"], z["orig"], [str(k) for k in z["keys"]]
    eu = np.sqrt(((soft - orig) ** 2).sum(1))
    for i, k in enumerate(keys):
        m = META[k]
        if i == 0:
            img = None
        # 直接读图（缓存的是画布，这里要原图）
        p = "%s/%s.jpg" % (k.split("/")[0], k.split("/")[1])
        im = C.imread_u("%s/%s/%s.jpg" % (C.ROOT, k.split("/")[0], k.split("/")[1]))
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
        H, W = g.shape
        gx, gy = gt[k]
        cx, cy = int(gx), int(gy)
        r = int(0.04 * W)                       # 中心凹附近窗口
        x0, x1 = max(0, cx - r), min(W, cx + r)
        y0, y1 = max(0, cy - r), min(H, cy + r)
        win = g[y0:y1, x0:x1]
        # 全图背景亮度（大核模糊）在该处的水平
        bg = cv2.blur(g, (201, 201))[cy, cx]
        # 到视野边缘的归一化距离（边缘更暗）
        ed = np.hypot((gx - m["cx"]) / m["r"], (gy - m["cy"]) / m["r"])
        rows.append(dict(key=k, err=eu[i], ecc=ed,
                         loc_mean=win.mean(), loc_std=win.std(),
                         bg=float(bg),
                         rel_contrast=float(win.std() / (win.mean() + 1e-6)),
                         edge_darkness=float(bg / (g.mean() + 1e-6))))
df = pd.DataFrame(rows)
print("n=%d" % len(df))
print()
print("误差与各图像属性的 Spearman 相关:")
for c in ["ecc", "loc_mean", "loc_std", "bg", "rel_contrast", "edge_darkness"]:
    rho = df[["err", c]].corr(method="spearman").iloc[0, 1]
    print("   %-16s rho=%+.3f   %s" % (c, rho,
          "*** 强" if abs(rho) > 0.35 else ("* 中" if abs(rho) > 0.2 else "")))
print()
# 多变量视角: 把 ecc 固定后, 亮度还有没有独立作用?
df["eccbin"] = pd.qcut(df.ecc, 3, labels=["近", "中", "远"])
print("按离心度分层后，局部亮度与误差的相关（检验亮度是否有独立作用）:")
for b, g in df.groupby("eccbin", observed=True):
    rho = g[["err", "loc_mean"]].corr(method="spearman").iloc[0, 1]
    rho2 = g[["err", "loc_std"]].corr(method="spearman").iloc[0, 1]
    print("   ecc=%-3s n=%2d  局部均值 rho=%+.3f   局部标准差 rho=%+.3f" % (b, len(g), rho, rho2))
print()
print("按局部亮度三分位看误差:")
df["brightbin"] = pd.qcut(df.loc_mean, 3, labels=["最暗", "中", "最亮"])
for b, g in df.groupby("brightbin", observed=True):
    print("   %-4s (n=%2d): 局部均值=%.0f~%.0f  误差 mean=%6.2f med=%5.2f p90=%6.2f  ecc中位=%.2f"
          % (b, len(g), g.loc_mean.min(), g.loc_mean.max(),
             g.err.mean(), g.err.median(), g.err.quantile(.9), g.ecc.median()))