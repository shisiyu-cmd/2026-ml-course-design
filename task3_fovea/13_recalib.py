# -*- coding: utf-8 -*-
"""【已作废 —— 结论是错的，保留仅供追溯】

!!!! 警告：本脚本的评估方式是错的 !!!!
它把拟合出来的线性式代入了「真值」G[i] 去计算误差，等于用答案反推预测，
属于数据泄漏。因此打印出的 "重标定后 RMSE≈8" 是假象。

正确的做法是把修正作用在「预测值」上，实测结果：
    正向拟合求逆  -> RMSE 28.47  (比原始 26.95 更差)
    反向回归      -> RMSE 26.75  (几乎无增益，在噪声范围内)
结论：观测到的收缩斜率(0.93/0.83)是「回归稀释」现象
      (pred = gt + 噪声 时最小二乘斜率天然 < 1)，不是可修正的系统偏差。

正确的根因排查与结论见 14_rootcause.py。
=============================================================================

原始说明（保留）：
验证「向圆心收缩」的系统偏差是否真实：
1) 逐模型重标定前后的 OOF RMSE
2) 重标定后最差样本的可视化核查
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

p = argparse.ArgumentParser()
p.add_argument("--tags", nargs="+", default=["base200", "seed1", "clahe1"])
p.add_argument("--variant", default="plain")
p.add_argument("--out", default="viz/recalib_check.png")
a = p.parse_args()

meta = C.load_meta()
D = {}
for t in a.tags:
    f = os.path.join("runs", "tta_%s.npz" % t)
    d = np.load(f, allow_pickle=True)
    D[t] = d["soft_" + a.variant]
    K = list(d["keys"]); GT = d["gt"]
M = np.array([[meta[k]["cx"], meta[k]["cy"], meta[k]["r"]] for k in K])
rel = lambda P: np.stack([(P[:, 0] - M[:, 0]) / M[:, 2], (P[:, 1] - M[:, 1]) / M[:, 2]], 1)
abs_ = lambda R: np.stack([M[:, 0] + R[:, 0] * M[:, 2], M[:, 1] + R[:, 1] * M[:, 2]], 1)
G = rel(GT)
rmse = lambda P: float(np.sqrt((((P - GT) ** 2)).mean()))

print("=" * 78)
print("逐模型：原始 RMSE  vs  逐轴线性重标定后的 RMSE (留一法)")
print("=" * 78)
print("%-12s %12s %12s %10s %10s" % ("tag", "原始", "重标定后", "X 斜率", "Y 斜率"))
fits = {}
for t in a.tags:
    R = rel(D[t])
    cx = np.polyfit(G[:, 0], R[:, 0], 1)
    cy = np.polyfit(G[:, 1], R[:, 1], 1)
    fits[t] = (cx, cy)
    # 留一法
    fx = np.array([np.polyval(np.polyfit(np.delete(G[:, 0], i), np.delete(R[:, 0], i), 1),
                              G[i, 0]) for i in range(len(G))])
    fy = np.array([np.polyval(np.polyfit(np.delete(G[:, 1], i), np.delete(R[:, 1], i), 1),
                              G[i, 1]) for i in range(len(G))])
    Pc = abs_(np.stack([fx, fy], 1))
    print("%-12s %12.3f %12.3f %10.4f %10.4f"
          % (t, rmse(D[t]), rmse(Pc), cx[0], cy[0]))

ens = np.mean([D[t] for t in a.tags], 0)
R = rel(ens)
cx = np.polyfit(G[:, 0], R[:, 0], 1)
cy = np.polyfit(G[:, 1], R[:, 1], 1)
fx = np.array([np.polyval(np.polyfit(np.delete(G[:, 0], i), np.delete(R[:, 0], i), 1),
                          G[i, 0]) for i in range(len(G))])
fy = np.array([np.polyval(np.polyfit(np.delete(G[:, 1], i), np.delete(R[:, 1], i), 1),
                          G[i, 1]) for i in range(len(G))])
Pc = abs_(np.stack([fx, fy], 1))
print("-" * 78)
print("%-12s %12.3f %12.3f %10.4f %10.4f"
      % ("%d模型集成" % len(a.tags), rmse(ens), rmse(Pc), cx[0], cy[0]))
print()
print("集成权重系数: X: pred = %.5f*gt + %.6f" % tuple(cx))
print("              Y: pred = %.5f*gt + %.6f" % tuple(cy))

# ---- 可视化: 重标定后最差的样本 ----
cache = C.build_cache(meta)
err0 = np.sqrt(((ens - GT) ** 2).sum(1))
err1 = np.sqrt(((Pc - GT) ** 2).sum(1))
order = np.argsort(-err1)[:8]
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for ax, i in zip(axes.ravel(), order):
    k = K[i]
    m = meta[k]
    im = cache[k]
    S = im.shape[0]
    gx, gy = C.to_canvas(m, *GT[i])
    ox, oy = C.to_canvas(m, *ens[i])
    rx, ry = C.to_canvas(m, *Pc[i])
    h = 200
    x0, x1 = max(0, int(gx) - h), min(S, int(gx) + h)
    y0, y1 = max(0, int(gy) - h), min(S, int(gy) + h)
    ax.imshow(im[y0:y1, x0:x1])
    ax.plot(gx - x0, gy - y0, "o", mfc="none", mec="lime", ms=20, mew=2)
    ax.plot(gx - x0, gy - y0, "+", c="lime", ms=13, mew=2)
    ax.plot(ox - x0, oy - y0, "x", c="red", ms=13, mew=2.5)
    ax.plot(rx - x0, ry - y0, "P", c="deepskyblue", ms=9, mew=2)
    ax.set_title("%s  原始=%.0fpx  重标定后=%.0fpx" % (k, err0[i], err1[i]), fontsize=9)
    ax.axis("off")
plt.suptitle("绿圈=真值  红X=原始预测  蓝P=重标定后预测", fontsize=14)
plt.tight_layout()
plt.savefig(a.out, dpi=78)
plt.close()
print("\n已保存", a.out)
print()
print("重标定后误差分布: p50=%.2f p75=%.2f p90=%.2f max=%.2f"
      % tuple(np.percentile(err1, [50, 75, 90, 100])))
print("原始误差分布:     p50=%.2f p75=%.2f p90=%.2f max=%.2f"
      % tuple(np.percentile(err0, [50, 75, 90, 100])))