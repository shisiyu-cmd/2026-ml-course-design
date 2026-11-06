# -*- coding: utf-8 -*-
"""核查外部数据（IDRiD）的裁剪与标注是否对齐。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

META = C.load_meta()
_, ext_gt = C.load_ext()
keys = sorted(ext_gt)
cache = C.build_cache(META, verbose=False)
print("ext 图像数: %d" % len(keys))

N = 8
sel = [keys[i] for i in np.linspace(0, len(keys) - 1, N).astype(int)]
fig, axes = plt.subplots(2, N, figsize=(3.0 * N, 6.6))
for j, k in enumerate(sel):
    img = C.imread_u("%s/%s/%s.jpg" % (C.ROOT, k.split("/")[0], k.split("/")[1]))
    cv = cache[k]
    m = META[k]
    gx, gy = ext_gt[k]
    tcx, tcy = C.to_canvas(m, gx, gy)
    # 原图 + 检测到的视野圆 + 真值
    axes[0, j].imshow(img)
    th = np.linspace(0, 2 * np.pi, 200)
    axes[0, j].plot(m["cx"] + m["r"] * np.cos(th), m["cy"] + m["r"] * np.sin(th), "c-", lw=1.2)
    axes[0, j].plot(gx, gy, "g+", ms=14, mew=2.5)
    axes[0, j].set_title("%s  (%dx%d)" % (k.split("/")[1], m["W"], m["H"]), fontsize=8)
    axes[0, j].axis("off")
    # 画布 + 真值
    axes[1, j].imshow(cv)
    axes[1, j].plot(tcx, tcy, "g+", ms=16, mew=3)
    axes[1, j].set_title("画布 真值=(%.0f,%.0f)" % (tcx, tcy), fontsize=8)
    axes[1, j].axis("off")
plt.suptitle("外部数据核查: IDRiD 视野圆检测(青) 与 黄斑真值(绿)", fontsize=13)
plt.tight_layout()
plt.savefig("viz/ext_check.png", dpi=72)
print("已保存 viz/ext_check.png")

# 定量: 真值是否落在画布内且有合理边距
bad = []
for k in keys:
    m = META[k]
    gx, gy = ext_gt[k]
    cx, cy = C.to_canvas(m, gx, gy)
    S = C.CANVAS
    if not (0.02 * S < cx < 0.98 * S and 0.02 * S < cy < 0.98 * S):
        bad.append((k, cx, cy))
print("真值落在画布边缘 2%% 以内的: %d 张" % len(bad))
for b in bad[:6]:
    print("   ", b[0], "(%.0f,%.0f)" % (b[1], b[2]))