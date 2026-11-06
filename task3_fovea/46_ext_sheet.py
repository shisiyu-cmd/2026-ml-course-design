# -*- coding: utf-8 -*-
"""检查外部数据(IDRiD)的标注质量：贴图 + 统计异常。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

META = C.load_meta()
_, EXT = C.load_ext()
KEYS = sorted(EXT)
print("外部数据: %d 张" % len(KEYS))

# 统计: 离心度分布, 到画布中心的距离
ecc, cen = [], []
for k in KEYS:
    m = META[k]
    gx, gy = EXT[k]
    cx, cy = C.to_canvas(m, gx, gy)
    ecc.append(np.hypot(gx - m["cx"], gy - m["cy"]) / m["r"])
    cen.append(np.hypot(cx - C.CANVAS / 2, cy - C.CANVAS / 2) / C.CANVAS)
ecc, cen = np.array(ecc), np.array(cen)
print("离心度:   中位 %.3f  p90 %.3f  max %.3f" % (np.median(ecc), np.percentile(ecc, 90), ecc.max()))
print("离画布中心: 中位 %.3f  p90 %.3f  max %.3f" % (np.median(cen), np.percentile(cen, 90), cen.max()))
print("离心度 >0.5r 的: %d 张" % int((ecc > 0.5).sum()))

# 均匀抽样 20 张贴图
sel = [KEYS[i] for i in np.linspace(0, len(KEYS) - 1, 20).astype(int)]
fig, axes = plt.subplots(4, 5, figsize=(21, 18))
for i, k in enumerate(sel):
    ax = axes[i // 5, i % 5]
    img = C.imread_u("%s/%s/%s.jpg" % (C.ROOT, k.split("/")[0], k.split("/")[1]))
    m = META[k]
    gx, gy = EXT[k]
    cx, cy = C.to_canvas(m, gx, gy)
    half = int(0.75 * m["r"])
    H, W = img.shape[:2]
    x0 = int(np.clip(gx - half, 0, max(0, W - 2 * half)))
    y0 = int(np.clip(gy - half, 0, max(0, H - 2 * half)))
    ax.imshow(img[y0:y0 + 2 * half, x0:x0 + 2 * half])
    ax.plot(gx - x0, gy - y0, "+", c="lime", ms=17, mew=3.4)
    ax.set_title("%s  ecc=%.2f" % (k.split("/")[1], np.hypot(gx - m["cx"], gy - m["cy"]) / m["r"]),
                 fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("viz/ext_sheet.png", dpi=62)
print("已保存 viz/ext_sheet.png")

# 列出离心度最大的 8 张（这些最可能有问题）
print()
print("离心度最大的 8 张（重点核查）:")
order = np.argsort(-ecc)[:8]
for j in order:
    print("   %-14s ecc=%.3f" % (KEYS[j], ecc[j]))