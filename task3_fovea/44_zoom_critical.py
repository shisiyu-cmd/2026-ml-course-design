# -*- coding: utf-8 -*-
"""高倍放大 0081 / 0086，用多种视图判读两个预测哪个更合理。"""
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

META = C.load_meta()
V4 = pd.read_csv("runs/test_pred_alldata.csv").set_index("key")
E5 = pd.read_csv("runs/test_pred_ext5.csv").set_index("key")
EX = pd.read_csv("runs/test_pred_allExt.csv").set_index("key")

KEYS = ["test/0081", "test/0086"]
HALF = 260          # 裁剪半宽（原图像素）

fig, axes = plt.subplots(len(KEYS) * 2, 4, figsize=(21, 5.4 * len(KEYS) * 2))
for ri, k in enumerate(KEYS):
    img = C.imread_u("%s/%s/%s.jpg" % (C.ROOT, k.split("/")[0], k.split("/")[1]))
    m = META[k]
    p4, p5, pe = (V4.x[k], V4.y[k]), (E5.x[k], E5.y[k]), (EX.x[k], EX.y[k])
    mx, my = (p4[0] + p5[0]) / 2, (p4[1] + p5[1]) / 2
    x0, y0 = int(mx - HALF), int(my - HALF)
    sub = img[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF]
    g = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY)

    views = [("原图", sub, None)]
    lab = cv2.cvtColor(sub, cv2.COLOR_RGB2LAB)
    for clip in (2.0, 6.0):
        l2 = lab.copy()
        l2[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l2[:, :, 0])
        views.append(("CLAHE %.1f" % clip, cv2.cvtColor(l2, cv2.COLOR_LAB2RGB), None))
    views.append(("绿通道", cv2.cvtColor(sub[:, :, 1], cv2.COLOR_GRAY2RGB), None))

    for vi, (nm, view, _) in enumerate(views):
        ax = axes[ri * 2, vi]
        ax.imshow(view)
        for (px, py), col, mk, lb in [(p4, "lime", "o", "v4"), (p5, "red", "x", "ext5"),
                                      (pe, "cyan", "+", "allExt")]:
            ax.plot(px - x0, py - y0, mk, c=col, ms=13, mew=2.8, label=lb)
        ax.set_title("%s  %s" % (k.split("/")[1], nm), fontsize=11)
        ax.axis("off")
        if ri == 0 and vi == 0:
            ax.legend(fontsize=9, loc="upper right")

    # 第二行：误差椭圆/差异可视化 + 距离
    ax = axes[ri * 2 + 1, 0]
    ax.imshow(sub)
    for (px, py), col, mk in [(p4, "lime", "o"), (p5, "red", "x"), (pe, "cyan", "+")]:
        ax.plot(px - x0, py - y0, mk, c=col, ms=13, mew=2.8)
    ax.plot([p4[0]-x0, p5[0]-x0], [p4[1]-y0, p5[1]-y0], "-", c="white", lw=1.4)
    d45 = float(np.hypot(p4[0]-p5[0], p4[1]-p5[1]))
    ax.set_title("三者位置   v4↔ext5=%.0f px" % d45, fontsize=11)
    ax.axis("off")

    # 视野圆相对位置（解剖学检查）
    ax = axes[ri * 2 + 1, 1]
    ax.imshow(sub)
    for rr in [0.25, 0.30, 0.35]:
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(m["cx"] + rr*m["r"]*np.cos(th) - x0, m["cy"] + rr*m["r"]*np.sin(th) - y0,
                "--", color="yellow", lw=1.0, alpha=0.8)
    for (px, py), col, mk in [(p4, "lime", "o"), (p5, "red", "x")]:
        ax.plot(px - x0, py - y0, mk, c=col, ms=13, mew=2.8)
    ax.plot(m["cx"]-x0, m["cy"]-y0, "w*", ms=15)
    ax.set_title("视盘(白星)与 0.25/0.30/0.35r 等距线", fontsize=10)
    ax.axis("off")

    # 绿通道放大（中心凹是血管弓内的无血管区）
    ax = axes[ri * 2 + 1, 2]
    gc = cv2.cvtColor(sub[:, :, 1], cv2.COLOR_GRAY2RGB)
    ax.imshow(gc)
    for (px, py), col, mk in [(p4, "lime", "o"), (p5, "red", "x")]:
        ax.plot(px - x0, py - y0, mk, c=col, ms=13, mew=2.8)
    ax.set_title("绿通道（无血管区）", fontsize=11)
    ax.axis("off")

    # 局部对比度热图
    ax = axes[ri * 2 + 1, 3]
    blur = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 12)
    ax.imshow(g.astype(np.float32) - blur, cmap="RdBu_r")
    for (px, py), col, mk in [(p4, "lime", "o"), (p5, "red", "x")]:
        ax.plot(px - x0, py - y0, mk, c=col, ms=13, mew=2.8)
    ax.set_title("高通（暗=比周围更暗）", fontsize=11)
    ax.axis("off")

    print("%s: v4=%s ext5=%s allExt=%s" % (k, np.round(p4), np.round(p5), np.round(pe)))
    print("   距视野中心: v4=%.3fr  ext5=%.3fr  allExt=%.3fr"
          % (np.hypot(p4[0]-m["cx"], p4[1]-m["cy"])/m["r"],
             np.hypot(p5[0]-m["cx"], p5[1]-m["cy"])/m["r"],
             np.hypot(pe[0]-m["cx"], pe[1]-m["cy"])/m["r"]))

plt.tight_layout()
plt.savefig("viz/zoom_0081_0086.png", dpi=62)
print("已保存 viz/zoom_0081_0086.png")